"""
研究者觀察工具：探針上下文、方言分散度、漂移
==========================================
policy probe：在一組固定的上下文下，讀出每個 agent 的行動分布 (A, n_ctx, n_actions)。
  * 方言分散度（between-agent dispersion）= agent 兩兩之間的平均 Jensen–Shannon 散度
    → 0 表示所有 agent 說同一種「話」；愈大表示個體分化愈大。
  * 漂移（drift）= 各 agent 與基礎模型（BC）分布的平均 JS。
"""
from __future__ import annotations

from typing import List

import numpy as np

from . import vocab as V


def _H(p, axis=-1):
    p = np.clip(p, 1e-12, 1)
    return -(p * np.log(p)).sum(axis)


def js(p, q):
    m = 0.5 * (p + q)
    return 0.5 * (_H(m) - _H(p)) + 0.5 * (_H(m) - _H(q))


def dispersion(probes: np.ndarray) -> float:
    """probes (A, n_ctx, K) → 所有 agent 兩兩之間 JS 的平均（跨上下文平均，nats）。
    與 drift 使用同一個 JS 定義，數值可直接比較。"""
    A = probes.shape[0]
    if A < 2:
        return 0.0
    i, j = np.triu_indices(A, 1)
    return float(js(probes[i], probes[j]).mean())


def drift(probes: np.ndarray, base: np.ndarray) -> float:
    """base (n_ctx, K)：基礎模型分布。"""
    return float(js(probes, base[None]).mean())


def make_probes(episodes: List[List[int]], n: int = 96, block: int = 16, seed: int = 0) -> np.ndarray:
    """從回合中取 n 個長度 block、以感知詞結尾的上下文；依結尾感知詞分層抽樣。"""
    rng = np.random.default_rng(seed)
    by_obs = {}
    for e in episodes:
        for end in range(block, len(e), 2):           # 交錯格式：奇數位置為感知詞
            if V.OBS_MASK[e[end - 1]]:
                by_obs.setdefault(e[end - 1], []).append(e[end - block:end])
    keys = sorted(by_obs)
    out = []
    while len(out) < n:
        for k in keys:
            if len(out) >= n:
                break
            pool = by_obs[k]
            out.append(pool[rng.integers(len(pool))])
    return np.array(out)
