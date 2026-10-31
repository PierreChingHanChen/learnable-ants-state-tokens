"""
GPTCore2：bo／dual 兩種模式的 microGPT 螞蟻
==========================================
bo   ：看 (o a)*，只預測行動。
dual ：看 (o s a)*；每一步先「說出」自己的控制狀態 s（從 STATE 詞取樣），再預測行動 a。
線上學習沿用 CwC 的篩選式自我模仿；dual 模式同時學 s 與 a（狀態是模型自己說的）。
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from ..microgpt import MicroGPT, GPTConfig
from . import vocab as V


class GPTCore2:
    kind = "gpt"

    def __init__(self, n_agents: int, rng: np.random.Generator, mode: str, base: Optional[MicroGPT] = None,
                 ctx_steps: int = 16, weight_mode: str = "individual", temperature: float = 1.0,
                 lr: float = 3e-3, batch: int = 16, n_updates: int = 4, social: bool = False, seed: int = 0,
                 n_embd: int = 16, learn_states: bool = True, weighting: str = "trip", sig_frac: float = 0.5,
                 rehearse: float = 0.0, rehearse_pool: Optional[List[List[int]]] = None):
        if mode not in V.MODES:
            raise ValueError(mode)
        self.A, self.rng, self.mode = n_agents, rng, mode
        self.weight_mode, self.temperature = weight_mode, temperature
        self.lr, self.batch, self.n_updates, self.social = lr, batch, n_updates, social
        nw = n_agents if weight_mode == "individual" else 1
        if base is None:
            cfg = GPTConfig(vocab_size=V.V, block_size=V.block_for(mode, ctx_steps), n_embd=n_embd)
            self.model = MicroGPT(cfg, n_agents=nw, seed=seed)
        else:
            self.model = MicroGPT.from_single(base.cfg, base.p, nw)
        self.block = self.model.cfg.block_size
        self.tmask = V.loss_mask(mode) if learn_states else V.ACTION_MASK   # 線上學習是否也學狀態詞
        self.learn_states = learn_states
        # 取樣加權（'trip' 預設／'step'／'sigma'）與教師回合複習（rehearse = 每批中教師窗口的比例）
        self.weighting, self.sig_frac = weighting, sig_frac
        self.rehearse = rehearse
        self.pool_teacher = None
        if rehearse > 0:
            pool = rehearse_pool or []
            self.pool_teacher = [V.to_bo(e) for e in pool] if mode == "bo" else pool
        self.ctx_steps = ctx_steps
        self.practiced: List[float] = []          # 每次學習時被練習的狀態目標中 σ 的比例

    def _w(self, ids):
        ids = np.asarray(ids)
        return ids if self.weight_mode == "individual" else np.zeros_like(ids)

    def _model_ctx(self, c: List[int]) -> List[int]:
        if self.mode == "aux":                         # 系統以 dual 格式記錄；aux 模型只看 o a
            return V.to_bo(c[-(3 * self.ctx_steps + 3):])[-self.block:]
        return c[-self.block:]

    def sample_aux(self, ids, contexts: List[List[int]]):
        """aux：同一次前向同時讀出狀態（不回饋）與行動。"""
        ids = np.asarray(ids)
        st = np.empty(len(ids), dtype=int)
        act = np.empty(len(ids), dtype=int)
        ctxs = [self._model_ctx(c) for c in contexts]
        lens = np.array([len(c) for c in ctxs])
        for L in np.unique(lens):
            sel = np.flatnonzero(lens == L)
            ctx = np.array([ctxs[i] for i in sel])
            for out, allowed in ((st, V.STATE_MASK), (act, V.ACTION_MASK)):
                p = self.model.next_token_probs(ctx, self._w(ids[sel]), allowed, self.temperature)
                u = self.rng.random((len(sel), 1))
                out[sel] = np.minimum((u > p.cumsum(1)).sum(1), V.V - 1)
        return st, act

    def sample(self, ids, contexts: List[List[int]], allowed: np.ndarray) -> np.ndarray:
        ids = np.asarray(ids)
        out = np.empty(len(ids), dtype=int)
        ctxs = [self._model_ctx(c) for c in contexts]
        lens = np.array([len(c) for c in ctxs])
        for L in np.unique(lens):
            sel = np.flatnonzero(lens == L)
            ctx = np.array([ctxs[i] for i in sel])
            p = self.model.next_token_probs(ctx, self._w(ids[sel]), allowed, self.temperature)
            u = self.rng.random((len(sel), 1))
            out[sel] = np.minimum((u > p.cumsum(1)).sum(1), V.V - 1)
        assert allowed[out].all()
        return out

    def _windows(self, seqs, n):
        """預設設定（trip、無複習、bo/dual）沿用 training_windows（亂數序列與早期版本相同）；其餘走 sample_windows。"""
        legacy = self.weighting == "trip" and self.pool_teacher is None and self.mode != "aux"
        if legacy:
            idx, tgt, msk = V.training_windows(seqs, n, self.block, self.rng, self.tmask)
            sm = msk & V.STATE_MASK[tgt]
            pr = float((tgt[sm] == V.SIGMA_ID).mean()) if sm.any() else float("nan")
            return idx, [(tgt, msk, None)], pr
        n_t = int(round(self.rehearse * n)) if self.pool_teacher is not None else 0
        idx, heads, pr = V.sample_windows(seqs, n - n_t, self.block, self.rng, self.mode, self.weighting,
                                          self.sig_frac, self.learn_states)
        if n_t:
            idx2, heads2, _ = V.sample_windows(self.pool_teacher, n_t, self.block, self.rng, self.mode, "trip",
                                               self.sig_frac, self.learn_states)
            idx = np.concatenate([idx, idx2])
            heads = [(np.concatenate([a[0], b[0]]), np.concatenate([a[1], b[1]]), a[2]) for a, b in zip(heads, heads2)]
            st_t = np.concatenate([h[0][h[1] & V.STATE_MASK[h[0]]] for h in heads]) if heads else np.array([])
            pr = float((st_t == V.SIGMA_ID).mean()) if len(st_t) else float("nan")
        return idx, heads, pr

    def _train(self, idx, heads, ids):
        if heads[0][2] is None:
            return self.model.train_step(idx, heads[0][0], heads[0][1], ids, self.lr)
        return self.model.train_step_heads(idx, heads, ids, self.lr)

    def learn(self, ids, buffers: Dict[int, List[List[int]]]) -> Dict:
        ids = [int(k) for k in ids if buffers.get(int(k))]
        if not ids:
            return {"n_agents": 0, "loss": float("nan")}
        pool = [s for k in buffers for s in buffers[k]] if (self.weight_mode == "shared" or self.social) else None
        losses, prs = [], []
        for _ in range(self.n_updates):
            if self.weight_mode == "shared":
                idx, heads, pr = self._windows(pool, self.batch * len(ids))
                loss = self._train(idx[None], [(h[0][None], h[1][None], h[2]) for h in heads], np.array([0]))
                prs.append(pr)
            else:
                packs = [self._windows(pool if self.social else buffers[k], self.batch) for k in ids]
                idx = np.stack([p[0] for p in packs])
                heads = [(np.stack([p[1][j][0] for p in packs]), np.stack([p[1][j][1] for p in packs]),
                          packs[0][1][j][2]) for j in range(len(packs[0][1]))]
                loss = self._train(idx, heads, np.array(ids))
                prs += [p[2] for p in packs]
            losses.append(float(np.mean(loss)))
        pr = float(np.nanmean(prs)) if np.isfinite(prs).any() else float("nan")
        self.practiced.append(pr)
        return {"n_agents": len(ids), "loss": float(np.mean(losses)), "practiced_sigma": pr}

    def policy_probe(self, contexts: np.ndarray) -> np.ndarray:
        """探針上下文（以感知詞〔bo、aux〕或狀態詞〔dual〕結尾）→ (n_weights, n_ctx, n_actions)。"""
        nw = self.model.A
        out = np.empty((nw, len(contexts), len(V.ACTION_IDS)))
        for w in range(nw):
            p = self.model.next_token_probs(contexts, np.full(len(contexts), w), V.ACTION_MASK, 1.0)
            out[w] = p[:, V.ACTION_IDS]
        return out

    def n_params(self) -> int:
        return self.model.n_params()
