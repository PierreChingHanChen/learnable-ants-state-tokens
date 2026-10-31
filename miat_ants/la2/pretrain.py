"""
LA2 行為複製預訓練
==================
collect()：以 EthoTeacher 在 two_patch 與 double_bridge 上執行，收集 dual 格式回合。
train()  ：bo 或 dual 模式訓練單一 microGPT（A=1），回傳模型與保留集指標。
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional

import numpy as np

from ..microgpt import MicroGPT, GPTConfig
from . import vocab as V
from .system import LA2Config, LA2System, base_path, WEIGHTS_DIR


def collect(seeds=range(500, 506), steps: int = 1500, maps=("two_patch", "double_bridge"),
            obs_flip: float = 0.0, teacher_kw: Optional[Dict] = None) -> List[List[int]]:
    eps = []
    for s in seeds:
        for m in maps:
            r = LA2System(LA2Config(map_name=m, core="teacher", seed=int(s), max_steps=steps,
                                    collect_episodes=True, obs_flip=obs_flip,
                                    teacher_kw=teacher_kw or {})).run()
            eps += r.episodes
    for e in eps[:50]:
        V.check_sequence(e, "dual")
    return eps


def split(eps, seed: int = 0, val_frac: float = 0.1):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(eps))
    n_val = max(1, int(len(eps) * val_frac))
    return [eps[i] for i in perm[n_val:]], [eps[i] for i in perm[:n_val]]


def as_mode(eps, mode):
    return [V.to_bo(e) for e in eps] if mode == "bo" else eps        # aux 由取樣器自行轉換


def evaluate(model: MicroGPT, eps_dual, mode: str, rng, n: int = 3000) -> Dict:
    if mode == "aux":
        return evaluate_aux(model, eps_dual, rng, n)
    eps = as_mode(eps_dual, mode)
    idx, tgt, _ = V.training_windows(eps, n, model.cfg.block_size, rng, V.loss_mask(mode))
    logits = model.forward(idx[None], np.array([0]))[0]
    out = {}
    for name, m in (("action", V.ACTION_MASK), ("state", V.STATE_MASK)):
        pos = m[tgt]
        if mode == "bo" and name == "state":
            continue
        lg = np.where(m[None, None, :], logits, -np.inf)
        lg = lg - lg.max(-1, keepdims=True)
        logp = lg - np.log(np.exp(lg).sum(-1, keepdims=True))
        nll = -np.take_along_axis(logp, tgt[..., None], -1)[..., 0][pos].mean()
        acc = (lg.argmax(-1) == tgt)[pos].mean()
        out[f"{name}_nll"], out[f"{name}_acc"] = float(nll), float(acc)
    return out


def evaluate_aux(model: MicroGPT, eps_dual, rng, n: int = 3000) -> Dict:
    idx, heads, _ = V.sample_windows(eps_dual, n, model.cfg.block_size, rng, "aux")
    logits = model.forward(idx[None], np.array([0]))[0]
    out = {}
    for name, (tgt, pos, m) in zip(("action", "state"), heads):
        lg = np.where(m[None, None, :], logits, -np.inf)
        lg = lg - lg.max(-1, keepdims=True)
        logp = lg - np.log(np.exp(lg).sum(-1, keepdims=True))
        out[f"{name}_nll"] = float(-np.take_along_axis(logp, tgt[..., None], -1)[..., 0][pos].mean())
        out[f"{name}_acc"] = float((lg.argmax(-1) == tgt)[pos].mean())
    return out


def train(mode: str, train_eps, val_eps, ctx_steps: int = 16, n_embd: int = 16, train_steps: int = 3000,
          batch: int = 64, lr: float = 1e-2, seed: int = 0, out: Optional[str] = None, verbose: bool = False):
    t0 = time.time()
    rng = np.random.default_rng(seed)
    tr = as_mode(train_eps, mode)
    cfg = GPTConfig(vocab_size=V.V, block_size=V.block_for(mode, ctx_steps), n_embd=n_embd)
    model = MicroGPT(cfg, n_agents=1, seed=seed)
    tm = V.loss_mask(mode)
    curve = []
    for s in range(train_steps):
        lr_s = lr * (1 - s / train_steps) + 1e-4
        if mode == "aux":
            idx, heads, _ = V.sample_windows(tr, batch, model.cfg.block_size, rng, "aux")
            loss = model.train_step_heads(idx[None], [(t[None], m[None], a) for t, m, a in heads], np.array([0]), lr_s)
        else:
            idx, tgt, msk = V.training_windows(tr, batch, model.cfg.block_size, rng, tm)
            loss = model.train_step(idx[None], tgt[None], msk[None], np.array([0]), lr_s)
        if s % 250 == 0 or s == train_steps - 1:
            curve.append((s, float(loss[0])))
            if verbose:
                print(f"  [{mode}] step {s:5d} loss {loss[0]:.4f}", flush=True)
    stats = evaluate(model, val_eps, mode, np.random.default_rng(seed + 1))
    stats.update({"mode": mode, "ctx_steps": ctx_steps, "n_embd": n_embd, "n_params": model.n_params(),
                  "train_eps": len(train_eps), "seconds": round(time.time() - t0, 1), "curve": curve})
    if out:
        os.makedirs(WEIGHTS_DIR, exist_ok=True)
        model.save(out)
        with open(out.replace(".npz", ".json"), "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=1)
    return model, stats
