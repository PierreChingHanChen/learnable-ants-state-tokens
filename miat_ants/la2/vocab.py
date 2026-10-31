"""
Learnable Ant 2 — 以文獻為本的雙字母語彙（Dual-Alphabet）
=========================================================
三類符號：
  感知詞 o：世界（World2.perceive）寫入
  控制狀態詞 s：教師的行為狀態（ethogram 狀態），dual 模式下由模型自己「說出」
  行動詞 a：核心產生
序列格式：
  bo   模式：.  o a  o a ...
  dual 模式：.  o s a  o s a ...
文獻依據見論文的 Methods 與補充材料第 S1 節。
"""
from __future__ import annotations

from typing import Iterable, List

import numpy as np

BOS = "."
# ---------------------------------------------------------------- 行動詞（10）
MOVE = ["F", "L", "R"]            # 前進／左轉45°前進／右轉45°前進
MOVE_MARK = ["f", "l", "r"]       # 同上並沉積費洛蒙
SPECIAL = ["U", "K"]              # U 形迴轉（Beckers 1992）／原地掃視轉向（Wystrach 2014）
OPS = ["T", "D"]                  # 拿取食物／在巢放下
ACTIONS = MOVE + MOVE_MARK + SPECIAL + OPS
# ---------------------------------------------------------------- 感知詞（25）
OBS_SEARCH = ["X", "Y", "A", "B", "C", "P", "Q", "S", "N", "W"]   # X 優質食物、Y 劣質食物（Beckers 1993）
OBS_GOAL = ["g", "h", "j", "k", "z"]          # 食物地點向量：前／右／後／左／已到（Bolek 2012）
OBS_CARRY = ["0", "w", "1", "2", "3", "4", "5", "6", "7", "8"]    # 1–4 巢方向（腳下路徑弱）、5–8（路徑強；Czaczkes 2013）
OBSERVATIONS = OBS_SEARCH + OBS_GOAL + OBS_CARRY
# ---------------------------------------------------------------- 控制狀態詞（8）
STATES = ["ε", "φ", "ν", "σ", "κ", "υ", "η", "δ"]
STATE_NAME = {"ε": "探索", "φ": "跟隨路徑", "ν": "向量導航", "σ": "系統搜尋",
              "κ": "掃視", "υ": "U 形迴轉", "η": "攜帶返巢", "δ": "卸貨"}
STATE_SOURCE = {"ε": "相關隨機漫步", "φ": "Draft et al. 2018", "ν": "Bolek et al. 2012",
                "σ": "Merkle et al. 2006", "κ": "Wystrach et al. 2014", "υ": "Beckers et al. 1992",
                "η": "Beckers 1993; Czaczkes 2013", "δ": "—"}

VOCAB: List[str] = [BOS] + ACTIONS + OBSERVATIONS + STATES
STOI = {s: i for i, s in enumerate(VOCAB)}
ITOS = {i: s for s, i in STOI.items()}
V = len(VOCAB)
BOS_ID = STOI[BOS]


def _mask(syms):
    m = np.zeros(V, dtype=bool)
    m[[STOI[s] for s in syms]] = True
    return m


ACTION_MASK = _mask(ACTIONS)
OBS_MASK = _mask(OBSERVATIONS)
STATE_MASK = _mask(STATES)
ACTION_IDS = np.flatnonzero(ACTION_MASK)
STATE_IDS = np.array([STOI[s] for s in STATES])
OBS_IDS = np.array([STOI[o] for o in OBSERVATIONS])
MARK_IDS = {STOI[a] for a in MOVE_MARK}
STATE_INDEX = {STOI[s]: i for i, s in enumerate(STATES)}     # token id → 0..7

MEANING = {
    ".": "回合開始", "F": "前進", "L": "左轉前進", "R": "右轉前進", "f": "前進＋標記", "l": "左轉前進＋標記",
    "r": "右轉前進＋標記", "U": "U 形迴轉", "K": "原地掃視轉向", "T": "拿取食物", "D": "放下食物",
    "X": "腳下優質食物", "Y": "腳下劣質食物", "A": "前方有食物", "B": "左前有食物", "C": "右前有食物",
    "P": "費洛蒙在前", "Q": "費洛蒙在左前", "S": "費洛蒙在右前", "N": "無線索", "W": "前方是牆",
    "g": "食物向量在前", "h": "食物向量在右", "j": "食物向量在後", "k": "食物向量在左", "z": "已到記憶地點（無食物）",
    "0": "在巢內", "w": "前方是牆（攜帶）", "1": "巢在前", "2": "巢在右", "3": "巢在後", "4": "巢在左",
    "5": "巢在前（強路徑）", "6": "巢在右（強路徑）", "7": "巢在後（強路徑）", "8": "巢在左（強路徑）",
    **{s: STATE_NAME[s] for s in STATES},
}

MODES = ("bo", "dual", "aux")     # aux：bo 格式輸入＋狀態讀出頭（監督狀態、但不回饋狀態）


def stride(mode: str) -> int:
    return 3 if mode == "dual" else 2


def block_for(mode: str, ctx_steps: int) -> int:
    return stride(mode) * ctx_steps


def encode(s: Iterable[str]) -> List[int]:
    return [STOI[c] for c in s]


def decode(ids: Iterable[int]) -> str:
    return "".join(ITOS[int(i)] for i in ids)


def to_bo(seq: List[int]) -> List[int]:
    """dual 序列 → bo 序列（去掉狀態詞）。"""
    return [t for t in seq if not STATE_MASK[t]]


def check_sequence(ids: List[int], mode: str) -> None:
    if not ids:
        return
    if ids[0] != BOS_ID:
        raise ValueError("序列必須以 BOS 開始")
    pat = [OBS_MASK, ACTION_MASK] if mode == "bo" else [OBS_MASK, STATE_MASK, ACTION_MASK]
    for k, t in enumerate(ids[1:]):
        if not pat[k % len(pat)][t]:
            raise ValueError(f"位置 {k + 1} 的 token {ITOS[t]!r} 不符合 {mode} 格式")


def training_windows(seqs: List[List[int]], n: int, block: int, rng: np.random.Generator,
                     target_mask: np.ndarray):
    """隨機取 n 個長度 block+1 的窗口 → (idx, targets, mask)；mask = 目標 token 屬於 target_mask。"""
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
        mask[i, :L] = target_mask[np.asarray(w[1:])]
    return idx, tgt, mask


def loss_mask(mode: str) -> np.ndarray:
    return (ACTION_MASK | STATE_MASK) if mode == "dual" else ACTION_MASK


# ---------------------------------------------------------------- 取樣加權與 aux 格式
SIGMA_ID = STOI["σ"]


def aux_view(seq: List[int]):
    """dual 序列 → (bo 序列, 狀態標籤)；狀態標籤與 bo 位置對齊：感知詞 o_t 的位置標 s_t，其餘為 -1。"""
    bo, lab = [], []
    for t in seq:
        if STATE_MASK[t]:
            lab[-1] = t
        else:
            bo.append(t)
            lab.append(-1)
    return bo, lab


def _pick_start(L: int, block: int, rng, p: int = -1, lo_off: int = 1) -> int:
    """在長度 L 的序列中選窗口起點；p ≥ 0 時窗口必須涵蓋位置 p（目標位置 st+lo_off … st+block-1+lo_off）。"""
    if L <= block + 1:
        return 0
    if p < 0:
        return int(rng.integers(0, L - block))
    lo = max(0, p - block + 1 - lo_off)
    hi = min(p - lo_off, L - block - 1)
    if hi < lo:
        return int(rng.integers(0, L - block))
    return int(rng.integers(lo, hi + 1))


def sample_windows(seqs: List[List[int]], n: int, block: int, rng: np.random.Generator, mode: str,
                   weighting: str = "trip", sig_frac: float = 0.5, learn_states: bool = True):
    """加權取樣器。seqs 為 dual 序列（bo 模式則為 bo 序列）。
    weighting：
      'trip'  —— 先均勻選回合、再選窗口（原設定：每趟等權，與 training_windows 相同）
      'step'  —— 依回合長度加權選回合（每一步大致等權）
      'sigma' —— 以機率 sig_frac 選一個含 σ 的回合，窗口必須涵蓋一個隨機的 σ 步；否則同 'trip'
    回傳：bo/dual → (idx, [(tgt, mask, None)]) ；aux → (idx, [(tgt_a, mask_a, ACTION_MASK), (tgt_s, mask_s, STATE_MASK)])
    另回傳 practiced：被練習的狀態目標中 σ 的比例（dual／aux）。"""
    stp = 2 if mode in ("bo", "aux") else 3
    lens = np.array([len(s) for s in seqs])
    ok = np.flatnonzero(lens >= 3)
    if weighting == "step":
        w = lens[ok].astype(float); w /= w.sum()
    sig_ok = None
    if weighting == "sigma":
        sig_ok = [i for i in ok if SIGMA_ID in seqs[i]]
    idx = np.full((n, block), BOS_ID, dtype=int)
    tgt = np.full((n, block), BOS_ID, dtype=int)
    mask = np.zeros((n, block), dtype=bool)
    tgs = np.full((n, block), STATE_IDS[0], dtype=int)
    mks = np.zeros((n, block), dtype=bool)
    tmask = loss_mask(mode) if (mode != "dual" or learn_states) else ACTION_MASK
    for i in range(n):
        p = -1
        if weighting == "sigma" and sig_ok and rng.random() < sig_frac:
            si = sig_ok[rng.integers(len(sig_ok))]
            s = seqs[si]
            sp = [j for j, t in enumerate(s) if t == SIGMA_ID]
            p = sp[rng.integers(len(sp))]
        elif weighting == "step":
            s = seqs[ok[rng.choice(len(ok), p=w)]]
        else:
            s = seqs[ok[rng.integers(len(ok))]]
        if mode == "aux":
            bo, lab = aux_view(s)
            if p >= 0:                                 # σ 在 dual 位置 p → 對應 bo 位置（其前面的感知詞）
                p = sum(1 for t in s[:p] if not STATE_MASK[t]) - 1
            st = _pick_start(len(bo), block, rng, p, lo_off=0)
            w_ = bo[st:st + block + 1]
            lb = lab[st:st + block]
            L = len(w_) - 1
            idx[i, :L] = w_[:-1]
            tgt[i, :L] = w_[1:]
            mask[i, :L] = ACTION_MASK[np.asarray(w_[1:])]
            lb = np.asarray(lb[:L])
            if learn_states:
                mks[i, :L] = lb >= 0
            tgs[i, :L] = np.where(lb >= 0, lb, STATE_IDS[0])
        else:
            st = _pick_start(len(s), block, rng, p, lo_off=1)
            w_ = s[st:st + block + 1]
            L = len(w_) - 1
            idx[i, :L] = w_[:-1]
            tgt[i, :L] = w_[1:]
            mask[i, :L] = tmask[np.asarray(w_[1:])]
    if mode == "aux":
        practiced = float((tgs[mks] == SIGMA_ID).mean()) if mks.any() else float("nan")
        return idx, [(tgt, mask, ACTION_MASK), (tgs, mks, STATE_MASK)], practiced
    sm = mask & STATE_MASK[tgt]
    practiced = float((tgt[sm] == SIGMA_ID).mean()) if sm.any() else float("nan")
    return idx, [(tgt, mask, None)], practiced
