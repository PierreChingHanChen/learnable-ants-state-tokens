"""
量測指標（Phase 4 量化驗證 + 三個研究主題的可操作化指標）
"""
from __future__ import annotations

from typing import Iterable, Sequence, Set, Tuple

import numpy as np


# ---------------------------------------------------------------- 原論文 式(9)(10)
def spatial_entropy(type_grid: np.ndarray, n_types: int, block: int = 8) -> float:
    """空間熵（原論文式 9–10，重建版）。

    將網格切成 block×block 的不重疊區塊 S；在每個非空區塊中計算
        E_S = -Σ_I P_I · log P_I ,   Σ_I P_I = 1
    其中 P_I 為物資類別 I 在區塊中的個數比率。以 log(n_types) 正規化到 [0,1]，
    再對所有非空區塊取平均。隨機分布時最大，聚類愈好愈低。

    type_grid: 整數陣列，-1 表示空格，0..n_types-1 為物資類別。
    """
    H, W = type_grid.shape
    ents = []
    for r in range(0, H, block):
        for c in range(0, W, block):
            blk = type_grid[r:r + block, c:c + block]
            counts = np.array([(blk == t).sum() for t in range(n_types)], dtype=float)
            tot = counts.sum()
            if tot == 0:
                continue
            p = counts[counts > 0] / tot
            ents.append(-(p * np.log(p)).sum() / np.log(n_types))
    return float(np.mean(ents)) if ents else 0.0


# ---------------------------------------------------------------- 主題一：We-ness
def trail_consensus(tau: np.ndarray) -> float:
    """路徑共識指數 C ∈ [0,1]：
    對每個節點 i，把 τ_i· 正規化成機率分布，計算熵 H_i；
        C = 1 - mean_i( H_i / log(n-1) )
    C≈0：介質沒有偏好（沒有「我們」的共同取向）；C→1：群體在介質上形成單一共識路徑。
    """
    n = tau.shape[0]
    t = tau.copy().astype(float)
    np.fill_diagonal(t, 0.0)
    p = t / t.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -np.nansum(np.where(p > 0, p * np.log(p), 0.0), axis=1)
    return float(1.0 - np.mean(h / np.log(n - 1)))


def stigmergic_gain(best_private: float, best_shared: float) -> float:
    """集體增益 G = L_private / L_shared（>1 表示共享介質帶來的增益）。"""
    return float(best_private / best_shared)


# ---------------------------------------------------------------- 主題二：責任
def gini(x: Sequence[float]) -> float:
    """Gini 係數：0＝責任平均分散（many hands），1＝集中於單一 agent。"""
    a = np.sort(np.asarray(x, dtype=float))
    if a.sum() == 0:
        return 0.0
    n = len(a)
    idx = np.arange(1, n + 1)
    return float((2 * (idx * a).sum()) / (n * a.sum()) - (n + 1) / n)


def effective_number(shares: Sequence[float]) -> float:
    """有效貢獻者數 N_eff = 1 / Σ s_k²（Herfindahl 倒數）。"""
    s = np.asarray(shares, dtype=float)
    s = s / s.sum()
    return float(1.0 / (s ** 2).sum())


# ---------------------------------------------------------------- 主題三：可逆性
def tour_edges(tour: Sequence[int]) -> Set[Tuple[int, int]]:
    t = list(tour)
    return {tuple(sorted((t[k], t[(k + 1) % len(t)]))) for k in range(len(t))}


def edge_jaccard(tour_a: Sequence[int], tour_b: Sequence[int]) -> float:
    """兩條巡迴路徑的邊集合 Jaccard 相似度（1＝同一條路徑）。"""
    a, b = tour_edges(tour_a), tour_edges(tour_b)
    return len(a & b) / len(a | b)


def recovery_time(series: Sequence[float], t_perturb: int, target: float,
                  tol: float = 0.02) -> int:
    """擾動後第一次回到 target·(1+tol) 以內所需的週期數；未恢復回傳 -1。"""
    s = np.asarray(series)
    for k in range(t_perturb, len(s)):
        if s[k] <= target * (1 + tol):
            return k - t_perturb
    return -1


def half_life(series: Sequence[float], t0: int) -> float:
    """量測某量（如被封鎖邊上的費洛蒙）自 t0 起衰減到一半所需週期數。"""
    s = np.asarray(series, dtype=float)
    v0 = s[t0]
    for k in range(t0, len(s)):
        if s[k] <= v0 / 2:
            return float(k - t0)
    return float("nan")
