"""
詞彙表與序列契約（F-01、F-02）
==============================
序列格式：  .  o₁ a₁  o₂ a₂  ...   （感知詞與行動詞交錯）
感知詞只能由世界（World.perceive）寫入；行動詞只能由核心（AgentCore.decide）產生。
"""
from __future__ import annotations

from typing import Iterable, List

import numpy as np

BOS = "."
# 行動詞
MOVE = ["F", "L", "R"]            # 前進／左轉45°前進／右轉45°前進
MOVE_MARK = ["f", "l", "r"]       # 同上並沉積費洛蒙
OPS = ["T", "D"]                  # 拿取食物／在巢放下
ACTIONS = MOVE + MOVE_MARK + OPS
# 感知詞：搜尋中
OBS_SEARCH = ["X", "A", "B", "C", "P", "Q", "S", "N", "W"]
# 感知詞：攜帶中
OBS_CARRY = ["0", "1", "2", "3", "4", "w"]
OBSERVATIONS = OBS_SEARCH + OBS_CARRY

VOCAB: List[str] = [BOS] + ACTIONS + OBSERVATIONS
STOI = {s: i for i, s in enumerate(VOCAB)}
ITOS = {i: s for s, i in STOI.items()}
V = len(VOCAB)

BOS_ID = STOI[BOS]
ACTION_IDS = np.array([STOI[a] for a in ACTIONS])
OBS_IDS = np.array([STOI[o] for o in OBSERVATIONS])
ACTION_MASK = np.zeros(V, dtype=bool)
ACTION_MASK[ACTION_IDS] = True
OBS_MASK = np.zeros(V, dtype=bool)
OBS_MASK[OBS_IDS] = True
MARK_IDS = {STOI[a] for a in MOVE_MARK}

MEANING = {
    ".": "回合開始", "F": "前進", "L": "左轉前進", "R": "右轉前進",
    "f": "前進＋標記", "l": "左轉前進＋標記", "r": "右轉前進＋標記", "T": "拿取食物", "D": "放下食物",
    "X": "腳下有食物", "A": "前方有食物", "B": "左前有食物", "C": "右前有食物",
    "P": "費洛蒙在前", "Q": "費洛蒙在左前", "S": "費洛蒙在右前", "N": "無線索", "W": "前方是牆",
    "0": "在巢內", "1": "巢在前", "2": "巢在右", "3": "巢在後", "4": "巢在左", "w": "前方是牆（攜帶中）",
}


def encode(s: Iterable[str]) -> List[int]:
    return [STOI[c] for c in s]


def decode(ids: Iterable[int]) -> str:
    return "".join(ITOS[int(i)] for i in ids)


def check_sequence(ids: List[int]) -> None:
    """驗證交錯格式：. 之後為 (感知, 行動) 交替。違反即拋出例外。"""
    if not ids:
        return
    if ids[0] != BOS_ID:
        raise ValueError("序列必須以 BOS 開始")
    for k, t in enumerate(ids[1:]):
        want_obs = (k % 2 == 0)
        if want_obs and not OBS_MASK[t]:
            raise ValueError(f"位置 {k + 1} 應為感知詞，得到 {ITOS[t]!r}")
        if (not want_obs) and not ACTION_MASK[t]:
            raise ValueError(f"位置 {k + 1} 應為行動詞，得到 {ITOS[t]!r}")


def training_windows(seqs: List[List[int]], n: int, block: int, rng: np.random.Generator):
    """從序列中隨機取 n 個長度 block+1 的窗口 → (idx, targets, mask)。
    mask 只在「目標為行動詞」的位置為 True。過短的序列以 BOS 左側補齊並遮罩。"""
    idx = np.full((n, block), BOS_ID, dtype=int)
    tgt = np.full((n, block), BOS_ID, dtype=int)
    mask = np.zeros((n, block), dtype=bool)
    lens = np.array([len(s) for s in seqs])
    ok = np.flatnonzero(lens >= 3)
    for i in range(n):
        s = seqs[ok[rng.integers(len(ok))]]
        if len(s) > block + 1:
            st = int(rng.integers(0, len(s) - block))
            w = s[st:st + block + 1]
        else:
            w = s
        L = len(w) - 1
        idx[i, :L] = w[:-1]
        tgt[i, :L] = w[1:]
        mask[i, :L] = ACTION_MASK[np.asarray(w[1:])]
    return idx, tgt, mask
