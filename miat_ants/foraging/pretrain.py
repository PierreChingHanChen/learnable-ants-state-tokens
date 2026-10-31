"""
A6 行為複製預訓練（離線）
========================
1. 以規則螞蟻（teacher）在兩種地圖上執行，收集所有回合的 token 序列。
2. 以單一 microGPT（A=1）在行動位置上做交叉熵訓練。
3. 量測保留集上的 NLL 與「最可能行動」和 teacher 行動的一致率，存成 base_bc.npz。

用法：python -m miat_ants.foraging.pretrain [--quick]
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

from ..microgpt import MicroGPT, GPTConfig
from . import vocab as V
from .system import ForagingConfig, ForagingSystem, BASE_WEIGHTS, WEIGHTS_DIR


def collect(n_runs: int = 6, steps: int = 1500, seed0: int = 500, mark_p: float = 1.0):
    eps = []
    for i in range(n_runs):
        for m in ("open_field", "double_bridge"):
            r = ForagingSystem(ForagingConfig(map_name=m, core="rule", seed=seed0 + i, max_steps=steps,
                                              collect_episodes=True, rule_mark_p=mark_p)).run()
            eps += r.episodes
    for e in eps:
        V.check_sequence(e)
    return eps


def evaluate(model: MicroGPT, eps, rng, n=2000):
    idx, tgt, msk = V.training_windows(eps, n, model.cfg.block_size, rng)
    logits = model.forward(idx[None], np.array([0]))[0]
    lg = np.where(V.ACTION_MASK[None, None, :], logits, -np.inf)
    lg -= lg.max(-1, keepdims=True)
    logp = lg - np.log(np.exp(lg).sum(-1, keepdims=True))
    nll = -np.take_along_axis(logp, tgt[..., None], -1)[..., 0][msk].mean()
    acc = (lg.argmax(-1) == tgt)[msk].mean()
    return float(nll), float(acc)


def pretrain(n_runs=6, steps=1500, train_steps=3000, batch=64, lr=1e-2, seed=0, out=BASE_WEIGHTS,
             verbose=True, mark_p=1.0):
    t0 = time.time()
    eps = collect(n_runs, steps, mark_p=mark_p)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(eps))
    n_val = max(1, len(eps) // 10)
    val = [eps[i] for i in perm[:n_val]]
    tr = [eps[i] for i in perm[n_val:]]
    model = MicroGPT(GPTConfig(vocab_size=V.V), n_agents=1, seed=seed)
    curve = []
    for s in range(train_steps):
        lr_s = lr * (1 - s / train_steps) + 1e-4
        idx, tgt, msk = V.training_windows(tr, batch, model.cfg.block_size, rng)
        loss = model.train_step(idx[None], tgt[None], msk[None], np.array([0]), lr_s)
        if s % 100 == 0 or s == train_steps - 1:
            curve.append((s, float(loss[0])))
            if verbose and s % 500 == 0:
                print(f"  step {s:5d}  loss {loss[0]:.4f}")
    nll, acc = evaluate(model, val, np.random.default_rng(seed + 1))
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    model.save(out)
    stats = {"episodes": len(eps), "tokens": int(sum(len(e) for e in eps)), "train_steps": train_steps,
             "val_nll": nll, "val_top1_agreement": acc, "n_params": model.n_params(),
             "seconds": round(time.time() - t0, 1), "curve": curve}
    with open(out.replace(".npz", ".json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)
    if verbose:
        print({k: v for k, v in stats.items() if k != "curve"})
    return model, stats


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    pretrain(n_runs=2 if quick else 6, steps=800 if quick else 1500, train_steps=600 if quick else 3000)
