"""
AgentCore：螞蟻的決策核心（A22 決策、A4 學習）
=============================================
介面契約
  decide(ids, contexts) -> action token ids
      contexts：每隻螞蟻最近 ≤ block_size 個 token（最後一個必為感知詞）。
      回傳值必屬於行動詞集合；不得讀取世界或介質（只能「讀」自己的文本）。
  learn(ids, buffers) -> dict
      只更新 ids 中各 agent 自己的參數。
  policy_probe(contexts) -> (A, n_ctx, n_actions)  研究者觀察介面

RuleCore：規則螞蟻（原生物模型；同時作為行為複製的 teacher）
GPTCore ：microGPT 螞蟻；weight_mode = 'individual'（共享基礎＋各自微調）或 'shared'（同一大腦）
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from ..microgpt import MicroGPT, GPTConfig
from . import vocab as V

S = V.STOI


class RuleCore:
    kind = "rule"

    def __init__(self, n_agents: int, rng: np.random.Generator, follow_p: float = 0.85,
                 noise: float = 0.1, mark_p: float = 1.0):
        self.A = n_agents
        self.rng = rng
        self.follow_p = follow_p
        self.noise = noise
        self.mark_p = mark_p          # 攜帶時以此機率標記（1.0 = 原始教師；不消耗亂數）

    def _rand(self, choices, p=None):
        return S[choices[self.rng.choice(len(choices), p=p)]]

    def decide(self, ids, contexts: List[List[int]]) -> np.ndarray:
        out = np.empty(len(ids), dtype=int)
        for i, ctx in enumerate(contexts):
            o = V.ITOS[ctx[-1]]
            if not V.OBS_MASK[ctx[-1]]:
                raise ValueError("上下文最後一個 token 必須是感知詞")
            if o == "X":
                a = S["T"]
            elif o == "W":
                a = self._rand(["L", "R"])
            elif o in "ABC":
                a = S[{"A": "F", "B": "L", "C": "R"}[o]]
            elif o in "PQS":
                a = S[{"P": "F", "Q": "L", "S": "R"}[o]] if self.rng.random() < self.follow_p \
                    else self._rand(["F", "L", "R"], [0.6, 0.2, 0.2])
            elif o == "N":
                a = self._rand(["F", "L", "R"], [0.7, 0.15, 0.15])
            elif o == "0":
                a = S["D"]
            elif o == "w":
                a = self._rand(["l", "r"])
            else:                                          # 攜帶中：朝巢走並標記
                if self.rng.random() < self.noise:
                    a = self._rand(["f", "l", "r"])
                else:
                    a = S[{"1": "f", "2": "r", "4": "l"}.get(o, "l" if self.rng.random() < 0.5 else "r")]
            if self.mark_p < 1.0 and V.ITOS[a] in "flr" and self.rng.random() >= self.mark_p:
                a = S[V.ITOS[a].upper()]                   # 不標記的同方向移動
            out[i] = a
        return out

    def learn(self, ids, buffers) -> Dict:
        return {}

    def n_params(self) -> int:
        return 0


class GPTCore:
    kind = "gpt"

    def __init__(self, n_agents: int, rng: np.random.Generator, base: Optional[MicroGPT] = None,
                 weight_mode: str = "individual", temperature: float = 1.0, lr: float = 3e-3,
                 batch: int = 16, n_updates: int = 4, social: bool = False, seed: int = 0):
        if weight_mode not in ("individual", "shared"):
            raise ValueError("weight_mode must be 'individual' or 'shared'")
        self.A = n_agents
        self.rng = rng
        self.weight_mode = weight_mode
        self.temperature = temperature
        self.lr, self.batch, self.n_updates, self.social = lr, batch, n_updates, social
        nw = n_agents if weight_mode == "individual" else 1
        if base is None:
            self.model = MicroGPT(GPTConfig(vocab_size=V.V), n_agents=nw, seed=seed)
        else:
            self.model = MicroGPT.from_single(base.cfg, base.p, nw)
        self.block = self.model.cfg.block_size

    def _w(self, ids):
        ids = np.asarray(ids)
        return ids if self.weight_mode == "individual" else np.zeros_like(ids)

    def decide(self, ids, contexts: List[List[int]]) -> np.ndarray:
        ids = np.asarray(ids)
        out = np.empty(len(ids), dtype=int)
        lens = np.array([min(len(c), self.block) for c in contexts])
        for L in np.unique(lens):
            sel = np.flatnonzero(lens == L)
            ctx = np.array([contexts[i][-L:] for i in sel])
            if not V.OBS_MASK[ctx[:, -1]].all():
                raise ValueError("上下文最後一個 token 必須是感知詞")
            p = self.model.next_token_probs(ctx, self._w(ids[sel]), V.ACTION_MASK, self.temperature)
            cum = p.cumsum(1)
            u = self.rng.random((len(sel), 1))
            out[sel] = np.minimum((u > cum).sum(1), V.V - 1)
        assert V.ACTION_MASK[out].all()
        return out

    def learn(self, ids, buffers: Dict[int, List[List[int]]]) -> Dict:
        """線上篩選式自我模仿：以各 agent 的成功回合做 n_updates 次 Adam 更新。"""
        ids = [int(k) for k in ids if buffers.get(int(k))]
        if not ids:
            return {"n_agents": 0, "loss": float("nan")}
        if self.weight_mode == "shared" or self.social:
            pool = [s for k in buffers for s in buffers[k]]
        losses = []
        for _ in range(self.n_updates):
            if self.weight_mode == "shared":
                idx, tgt, msk = V.training_windows(pool, self.batch * len(ids), self.block, self.rng)
                loss = self.model.train_step(idx[None], tgt[None], msk[None], np.array([0]), self.lr)
            else:
                packs = [V.training_windows(pool if self.social else buffers[k], self.batch, self.block, self.rng)
                         for k in ids]
                idx = np.stack([p[0] for p in packs]); tgt = np.stack([p[1] for p in packs])
                msk = np.stack([p[2] for p in packs])
                loss = self.model.train_step(idx, tgt, msk, np.array(ids), self.lr)
            losses.append(float(np.mean(loss)))
        return {"n_agents": len(ids), "loss": float(np.mean(losses))}

    def policy_probe(self, contexts: np.ndarray) -> np.ndarray:
        """固定探針上下文 (n_ctx, T) → (n_weights, n_ctx, n_actions) 行動分布。"""
        nw = self.model.A
        out = np.empty((nw, len(contexts), len(V.ACTION_IDS)))
        for w in range(nw):
            p = self.model.next_token_probs(contexts, np.full(len(contexts), w), V.ACTION_MASK, 1.0)
            out[w] = p[:, V.ACTION_IDS]
        return out

    def n_params(self) -> int:
        return self.model.n_params()
