"""
MicroGPT（numpy 版，多 agent 批次）
==================================

架構對齊 Karpathy microgpt（2026）：token／位置嵌入、RMSNorm、一層多頭因果自注意力、
ReLU MLP、殘差連接、語言模型輸出頭；預設 n_embd=16、n_head=4、block_size=16，
參數約 4 千。差異：
  * 以 numpy 張量實作並手寫反向傳播（原版為逐純量 autograd，太慢無法做多 agent 模擬）。
  * 所有參數帶有前導的 agent 維度 A：每個 agent 擁有自己的權重，
    但可在一次前向／反向中同時計算（對應 MIAT 的族群控制器）。
  * 損失只計算在 mask 為 True 的位置（行動位置）。

張量形狀：idx (A, B, T)；參數 wte (A,V,C)、wpe (A,Tmax,C)、wq/wk/wv/wo (A,C,C)、
w1 (A,C,4C)、w2 (A,4C,C)、lm (A,C,V)。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

EPS = 1e-5
PARAM_NAMES = ("wte", "wpe", "wq", "wk", "wv", "wo", "w1", "w2", "lm")


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 16
    n_embd: int = 16
    n_head: int = 4
    mlp_mult: int = 4


def _rms(x):
    r = 1.0 / np.sqrt((x * x).mean(-1, keepdims=True) + EPS)
    return x * r, r


def _rms_back(dn, n, r):
    """n = x·r，r = 1/sqrt(mean(x²)+eps)。回傳 dL/dx。"""
    return r * (dn - n * (dn * n).mean(-1, keepdims=True))


class MicroGPT:
    def __init__(self, cfg: GPTConfig, n_agents: int = 1, seed: int = 0,
                 params: Optional[Dict[str, np.ndarray]] = None):
        self.cfg = cfg
        self.A = n_agents
        if params is not None:
            self.p = {k: v.copy() for k, v in params.items()}
            self.A = self.p["wte"].shape[0]
        else:
            rng = np.random.default_rng(seed)
            V, C, T, M = cfg.vocab_size, cfg.n_embd, cfg.block_size, cfg.n_embd * cfg.mlp_mult
            std = 0.08
            one = {"wte": rng.normal(0, std, (V, C)), "wpe": rng.normal(0, std, (T, C)),
                   "wq": rng.normal(0, std, (C, C)), "wk": rng.normal(0, std, (C, C)),
                   "wv": rng.normal(0, std, (C, C)), "wo": np.zeros((C, C)),
                   "w1": rng.normal(0, std, (C, M)), "w2": np.zeros((M, C)),
                   "lm": rng.normal(0, std, (C, V))}
            self.p = {k: np.repeat(v[None], n_agents, axis=0) for k, v in one.items()}
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.t = np.zeros(self.A, dtype=int)

    # ------------------------------------------------------------ utilities
    def n_params(self) -> int:
        return int(sum(v[0].size for v in self.p.values()))

    def subset(self, ids) -> Dict[str, np.ndarray]:
        return {k: v[ids] for k, v in self.p.items()}

    @classmethod
    def from_single(cls, cfg: GPTConfig, single: Dict[str, np.ndarray], n_agents: int) -> "MicroGPT":
        """把一份（A=1）權重複製給 n 個 agent。"""
        return cls(cfg, params={k: np.repeat(v[:1], n_agents, axis=0) for k, v in single.items()})

    def save(self, path: str) -> None:
        np.savez_compressed(path, **self.p, _vocab=self.cfg.vocab_size, _block=self.cfg.block_size,
                            _embd=self.cfg.n_embd, _head=self.cfg.n_head, _mlp=self.cfg.mlp_mult)

    @classmethod
    def load(cls, path: str) -> "MicroGPT":
        z = np.load(path)
        cfg = GPTConfig(int(z["_vocab"]), int(z["_block"]), int(z["_embd"]), int(z["_head"]), int(z["_mlp"]))
        return cls(cfg, params={k: z[k] for k in PARAM_NAMES})

    # --------------------------------------------------------------- forward
    def forward(self, idx: np.ndarray, ids: Optional[np.ndarray] = None, cache: bool = False):
        """idx (A',B,T) → logits (A',B,T,V)。ids 指定使用哪些 agent 的權重（預設全部）。"""
        P = self.p if ids is None else self.subset(ids)
        A, B, T = idx.shape
        C, H = self.cfg.n_embd, self.cfg.n_head
        hd = C // H
        if T > self.cfg.block_size:
            raise ValueError("序列長度超過 block_size")
        ar = np.arange(A)[:, None, None]
        x0 = P["wte"][ar, idx] + P["wpe"][:, None, :T, :]
        h1, r1 = _rms(x0)
        q = h1 @ P["wq"][:, None]
        k = h1 @ P["wk"][:, None]
        v = h1 @ P["wv"][:, None]
        sp = lambda z: z.reshape(A, B, T, H, hd).transpose(0, 1, 3, 2, 4)
        qh, kh, vh = sp(q), sp(k), sp(v)
        scale = 1.0 / np.sqrt(hd)
        s = qh @ kh.transpose(0, 1, 2, 4, 3) * scale
        mask = np.triu(np.ones((T, T), dtype=bool), 1)
        s = np.where(mask, -1e9, s)
        s = s - s.max(-1, keepdims=True)
        pa = np.exp(s)
        pa /= pa.sum(-1, keepdims=True)
        yh = pa @ vh
        y = yh.transpose(0, 1, 3, 2, 4).reshape(A, B, T, C)
        o = y @ P["wo"][:, None]
        x1 = x0 + o
        h2, r2 = _rms(x1)
        u = h2 @ P["w1"][:, None]
        mlp = np.maximum(u, 0)
        f = mlp @ P["w2"][:, None]
        x2 = x1 + f
        h3, r3 = _rms(x2)
        logits = h3 @ P["lm"][:, None]
        if cache:
            self._c = dict(P=P, idx=idx, ar=ar, h1=h1, r1=r1, qh=qh, kh=kh, vh=vh, pa=pa, y=y, scale=scale,
                           h2=h2, r2=r2, u=u, mlp=mlp, h3=h3, r3=r3, shape=(A, B, T, C, H, hd))
        return logits


    def forward_last(self, idx: np.ndarray, ids: Optional[np.ndarray] = None) -> np.ndarray:
        """與 forward(idx)[:, :, -1, :] 相同，但只計算最後一個位置（取樣用，較快）。"""
        P = self.p if ids is None else self.subset(ids)
        A, B, T = idx.shape
        C, H = self.cfg.n_embd, self.cfg.n_head
        hd = C // H
        if T > self.cfg.block_size:
            raise ValueError("序列長度超過 block_size")
        ar = np.arange(A)[:, None, None]
        x0 = P["wte"][ar, idx] + P["wpe"][:, None, :T, :]
        h1, _ = _rms(x0)
        k = (h1 @ P["wk"][:, None]).reshape(A, B, T, H, hd).transpose(0, 1, 3, 2, 4)      # (A,B,H,T,hd)
        v = (h1 @ P["wv"][:, None]).reshape(A, B, T, H, hd).transpose(0, 1, 3, 2, 4)
        q = (h1[:, :, -1:, :] @ P["wq"][:, None]).reshape(A, B, 1, H, hd).transpose(0, 1, 3, 2, 4)
        s = q @ k.transpose(0, 1, 2, 4, 3) * (1.0 / np.sqrt(hd))                         # (A,B,H,1,T)
        s = s - s.max(-1, keepdims=True)
        pa = np.exp(s)
        pa /= pa.sum(-1, keepdims=True)
        y = (pa @ v).transpose(0, 1, 3, 2, 4).reshape(A, B, 1, C)
        x1 = x0[:, :, -1:, :] + y @ P["wo"][:, None]
        h2, _ = _rms(x1)
        x2 = x1 + np.maximum(h2 @ P["w1"][:, None], 0) @ P["w2"][:, None]
        h3, _ = _rms(x2)
        return (h3 @ P["lm"][:, None])[:, :, 0, :]

    @staticmethod
    def loss_restricted(logits, targets, mask, allowed):
        """只在 allowed 詞上做 softmax 的交叉熵（多頭讀出用）。回傳每個 agent 的損失總和 (A,)、計數 (A,) 與 dlogits（未正規化）。"""
        A = logits.shape[0]
        lg = np.where(allowed, logits, -np.inf)
        z = lg - lg.max(-1, keepdims=True)
        ez = np.where(allowed, np.exp(z), 0.0)
        sm = ez / ez.sum(-1, keepdims=True)
        tgt = np.where(mask, targets, int(np.flatnonzero(allowed)[0]))
        logp_t = np.log(np.take_along_axis(sm, tgt[..., None], -1)[..., 0] + 1e-300)
        cnt = mask.reshape(A, -1).sum(1).astype(float)
        lsum = -(logp_t * mask).reshape(A, -1).sum(1)
        d = sm.copy()
        np.put_along_axis(d, tgt[..., None], np.take_along_axis(d, tgt[..., None], -1) - 1, -1)
        d *= mask[..., None]
        return lsum, cnt, d

    def train_step_heads(self, idx, heads, ids=None, lr=1e-2) -> np.ndarray:
        """多頭訓練：heads = [(targets, mask, allowed), ...]，各頭在自己的詞群上做 softmax，
        損失 = 所有目標位置的平均（與 dual 模式「狀態與行動位置等權」一致）。"""
        logits = self.forward(idx, ids, cache=True)
        A = logits.shape[0]
        tot = np.zeros(A); cnt = np.zeros(A); dl = np.zeros_like(logits)
        for tg, mk, al in heads:
            l, c, d = self.loss_restricted(logits, tg, mk, al)
            tot += l; cnt += c; dl += d
        cnt = np.maximum(cnt, 1)
        dl /= cnt[:, None, None, None]
        grads = self.backward(dl)
        self.step(grads, ids, lr)
        return tot / cnt

    # ------------------------------------------------------------------ loss
    @staticmethod
    def loss_and_grad_logits(logits, targets, mask):
        """交叉熵（只計 mask 位置）。回傳每個 agent 的平均損失 (A,) 與 dlogits。"""
        A = logits.shape[0]
        z = logits - logits.max(-1, keepdims=True)
        ez = np.exp(z)
        sm = ez / ez.sum(-1, keepdims=True)
        logp = z - np.log(ez.sum(-1, keepdims=True))
        tgt_logp = np.take_along_axis(logp, targets[..., None], -1)[..., 0]
        cnt = np.maximum(mask.reshape(A, -1).sum(1), 1).astype(float)
        loss = -(tgt_logp * mask).reshape(A, -1).sum(1) / cnt
        d = sm.copy()
        np.put_along_axis(d, targets[..., None], np.take_along_axis(d, targets[..., None], -1) - 1, -1)
        d *= mask[..., None] / cnt[:, None, None, None]
        return loss, d

    # -------------------------------------------------------------- backward
    def backward(self, dlogits) -> Dict[str, np.ndarray]:
        c = self._c
        P = c["P"]
        A, B, T, C, H, hd = c["shape"]
        BT = B * T
        fl = lambda z: z.reshape(A, BT, z.shape[-1])
        wgrad = lambda a, b: fl(a).transpose(0, 2, 1) @ fl(b)          # Σ_bt a⊗b → (A, da, db)
        g = {}
        g["lm"] = wgrad(c["h3"], dlogits)
        dh3 = dlogits @ P["lm"].transpose(0, 2, 1)[:, None]
        dx2 = _rms_back(dh3, c["h3"], c["r3"])
        dx1 = dx2.copy()
        g["w2"] = wgrad(c["mlp"], dx2)
        dm = dx2 @ P["w2"].transpose(0, 2, 1)[:, None]
        du = dm * (c["u"] > 0)
        g["w1"] = wgrad(c["h2"], du)
        dh2 = du @ P["w1"].transpose(0, 2, 1)[:, None]
        dx1 += _rms_back(dh2, c["h2"], c["r2"])
        dx0 = dx1.copy()
        g["wo"] = wgrad(c["y"], dx1)
        dy = dx1 @ P["wo"].transpose(0, 2, 1)[:, None]
        dyh = dy.reshape(A, B, T, H, hd).transpose(0, 1, 3, 2, 4)
        pa = c["pa"]
        dpa = dyh @ c["vh"].transpose(0, 1, 2, 4, 3)
        dvh = pa.transpose(0, 1, 2, 4, 3) @ dyh
        ds = pa * (dpa - (dpa * pa).sum(-1, keepdims=True)) * c["scale"]
        dqh = ds @ c["kh"]
        dkh = ds.transpose(0, 1, 2, 4, 3) @ c["qh"]
        mg = lambda z: z.transpose(0, 1, 3, 2, 4).reshape(A, B, T, C)
        dq, dk, dv = mg(dqh), mg(dkh), mg(dvh)
        h1 = c["h1"]
        g["wq"] = wgrad(h1, dq)
        g["wk"] = wgrad(h1, dk)
        g["wv"] = wgrad(h1, dv)
        dh1 = (dq @ P["wq"].transpose(0, 2, 1)[:, None] + dk @ P["wk"].transpose(0, 2, 1)[:, None]
               + dv @ P["wv"].transpose(0, 2, 1)[:, None])
        dx0 += _rms_back(dh1, h1, c["r1"])
        g["wte"] = np.zeros_like(P["wte"])
        np.add.at(g["wte"], (np.broadcast_to(c["ar"], c["idx"].shape), c["idx"]), dx0)
        g["wpe"] = np.zeros_like(P["wpe"])
        g["wpe"][:, :T] = dx0.sum(1)
        return g

    # ------------------------------------------------------------------ Adam
    def step(self, grads: Dict[str, np.ndarray], ids: Optional[np.ndarray] = None, lr: float = 1e-2,
             b1: float = 0.9, b2: float = 0.99, clip: float = 1.0) -> None:
        """只更新 ids 指定的 agent（預設全部）。"""
        ids = np.arange(self.A) if ids is None else np.asarray(ids)
        # 每個 agent 各自做梯度裁剪
        norm = np.sqrt(sum((g.reshape(len(ids), -1) ** 2).sum(1) for g in grads.values()))
        sc = np.minimum(1.0, clip / (norm + 1e-12))
        self.t[ids] += 1
        t = self.t[ids].astype(float)
        for k, g in grads.items():
            g = g * sc.reshape((-1,) + (1,) * (g.ndim - 1))
            m = self.m[k][ids] = b1 * self.m[k][ids] + (1 - b1) * g
            v = self.v[k][ids] = b2 * self.v[k][ids] + (1 - b2) * g * g
            bc1 = (1 - b1 ** t).reshape((-1,) + (1,) * (g.ndim - 1))
            bc2 = (1 - b2 ** t).reshape((-1,) + (1,) * (g.ndim - 1))
            self.p[k][ids] -= lr * (m / bc1) / (np.sqrt(v / bc2) + 1e-8)

    def train_step(self, idx, targets, mask, ids=None, lr=1e-2) -> np.ndarray:
        logits = self.forward(idx, ids, cache=True)
        loss, dl = self.loss_and_grad_logits(logits, targets, mask)
        grads = self.backward(dl)
        self.step(grads, ids, lr)
        return loss

    # -------------------------------------------------------------- sampling
    def next_token_probs(self, contexts: np.ndarray, ids: np.ndarray, allowed: np.ndarray,
                         temperature: float = 1.0) -> np.ndarray:
        """contexts (n, T)，第 i 列使用 agent ids[i] 的權重 → (n, V) 機率，只在 allowed 上非零。"""
        logits = self.forward_last(contexts[:, None, :], ids)[:, 0, :] / max(temperature, 1e-6)
        logits = np.where(allowed[None, :], logits, -np.inf)
        logits -= logits.max(-1, keepdims=True)
        p = np.exp(logits)
        return p / p.sum(-1, keepdims=True)
