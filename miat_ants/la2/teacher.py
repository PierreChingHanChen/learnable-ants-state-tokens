"""
EthoTeacher：以文獻行為建構的規則教師（產生 o → s → a）
=====================================================
每隻螞蟻一個行為狀態 s ∈ {ε 探索, φ 跟隨路徑, ν 向量導航, σ 系統搜尋, κ 掃視, υ U 形迴轉, η 攜帶返巢, δ 卸貨}。
教師是資料的生成過程：它可讀取世界的特權資訊（到搜尋中心的距離、到達後經過步數），
但學習型螞蟻只看得到感知詞。轉移規則與參數：

  出巢        有食物地點向量 → ν；否則以 p_scan_start 機率先掃視 κ（新手幾乎每趟掃視，Wystrach 2014）→ ε
  ε 探索      相關隨機漫步（F .70 / L .15 / R .15）；感知到路徑 → φ
  φ 跟隨      以 follow_p 跟隨（Draft 2018）；失去路徑後再走 lost_keep 步，之後以 p_uturn 做 U 形迴轉 υ
              （Beckers 1992），否則回到 ε
  υ 迴轉      單步 U；下一步依感知回到 φ 或 ε
  ν 向量      依 g/h/k 前進／右轉／左轉；目標在後（j）時以 p_scan_conflict 掃視（線索衝突，Wystrach 2014）
              到達空的記憶地點（z）→ σ
  σ 搜尋      以記憶地點為中心、半徑 2 + t/8 的迴圈（Merkle 2006）；離中心超過半徑即折返
              世界在 give_up 步後遺忘向量 → ε
  η 返巢      依巢方向前進並標記；標記機率 = q(品質) × 0.5(強路徑) × 1.3(本趟曾搜尋)（Beckers 1993；Czaczkes 2013）
  δ 卸貨      在巢放下食物
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from . import vocab as V

S = V.STOI
SI = {s: S[s] for s in V.STATES}


class EthoTeacher:
    kind = "teacher"

    def __init__(self, n_agents: int, rng: np.random.Generator, world=None, follow_p: float = 0.85,
                 noise: float = 0.1, lost_keep: int = 2, p_uturn: float = 0.5, p_scan_start: float = 0.6,
                 p_scan_conflict: float = 0.5, scan_len: int = 2, q_mark=(0.7, 1.0), strong_factor: float = 0.5,
                 search_boost: float = 1.3):
        self.A = n_agents
        self.rng = rng
        self.world = world
        self.follow_p, self.noise = follow_p, noise
        self.lost_keep, self.p_uturn = lost_keep, p_uturn
        self.p_scan_start, self.p_scan_conflict, self.scan_len = p_scan_start, p_scan_conflict, scan_len
        self.q_mark, self.strong_factor, self.search_boost = q_mark, strong_factor, search_boost
        self.state = np.array(["ε"] * n_agents, dtype=object)
        self.lost = np.zeros(n_agents, int)
        self.scan_left = np.zeros(n_agents, int)
        self.after_scan = np.array(["ε"] * n_agents, dtype=object)
        self.searched = np.zeros(n_agents, bool)
        for k in range(n_agents):
            self.new_trip(k)

    # ------------------------------------------------------------ helpers
    def _r(self, choices, p=None):
        return choices[self.rng.choice(len(choices), p=p)]

    def new_trip(self, k: int) -> None:
        has_mem = bool(self.world.has_mem[k]) if self.world is not None else False
        self.searched[k] = False
        self.lost[k] = 0
        if has_mem:
            self.state[k] = "ν"
        elif self.rng.random() < self.p_scan_start:
            self.state[k], self.scan_left[k], self.after_scan[k] = "κ", self.scan_len, "ε"
        else:
            self.state[k] = "ε"

    def _crw(self):
        return self._r(["F", "L", "R"], [0.7, 0.15, 0.15])

    # ------------------------------------------------------------ decide
    def decide_full(self, ids, obs) -> (np.ndarray, np.ndarray):
        st_out = np.empty(len(ids), dtype=int)
        a_out = np.empty(len(ids), dtype=int)
        w = self.world
        for i, k in enumerate(ids):
            o = V.ITOS[int(obs[i])]
            s = self.state[k]
            a = None
            if w.carrying[k]:
                if o == "0":
                    s, a = "δ", "D"
                else:
                    s = "η"
                    if o == "w":
                        a = self._r(["l", "r"])
                    elif self.rng.random() < self.noise:
                        a = self._r(["f", "l", "r"])
                    else:
                        d = (int(o) - 1) % 4
                        a = ["f", "r", None, "l"][d] or self._r(["l", "r"])
                    pm = self.q_mark[int(w.carry_q[k])]
                    if o in "5678":
                        pm *= self.strong_factor
                    if self.searched[k]:
                        pm *= self.search_boost
                    if self.rng.random() >= min(pm, 1.0):
                        a = a.upper()
            else:
                if s in ("η", "δ"):                      # 不應發生（卸貨後 new_trip 會重設）
                    s = "ε"
                if s == "υ":
                    s = "φ" if o in "PQS" else "ε"
                if s in ("ν", "σ") and not w.has_mem[k]:  # 世界已遺忘向量
                    s = "φ" if o in "PQS" else "ε"
                if o in "XY":
                    a = "T"
                elif o == "W":
                    a = self._r(["L", "R"])
                    if s == "κ":
                        self.scan_left[k] -= 1
                elif o in "ABC":
                    a = {"A": "F", "B": "L", "C": "R"}[o]
                elif s == "κ":
                    a = "K"
                    self.scan_left[k] -= 1
                elif s == "ε":
                    if o in "PQS":
                        s = "φ"
                        a = {"P": "F", "Q": "L", "S": "R"}[o] if self.rng.random() < self.follow_p else self._crw()
                    else:
                        a = self._crw()
                elif s == "φ":
                    if o in "PQS":
                        self.lost[k] = 0
                        a = {"P": "F", "Q": "L", "S": "R"}[o] if self.rng.random() < self.follow_p else self._crw()
                    else:
                        self.lost[k] += 1
                        if self.lost[k] <= self.lost_keep:
                            a = "F"
                        elif self.rng.random() < self.p_uturn:
                            s, a = "υ", "U"
                            self.lost[k] = 0
                        else:
                            s, a = "ε", self._crw()
                            self.lost[k] = 0
                elif s == "ν":
                    if o == "z":
                        s = "σ"
                        self.searched[k] = True
                        a = self._r(["F", "L", "R"], [0.5, 0.25, 0.25])
                    elif o == "j" and self.rng.random() < self.p_scan_conflict:
                        s, a = "κ", "K"
                        self.scan_left[k], self.after_scan[k] = self.scan_len - 1, "ν"
                    elif self.rng.random() < self.noise:
                        a = self._crw()
                    else:
                        a = {"g": "F", "h": "R", "k": "L"}.get(o) or self._r(["L", "R"])
                elif s == "σ":
                    t = max(int(w.arrive_t[k]), 0)
                    radius = 2 + t // 8
                    if w.goal_dist(k) > radius:
                        a = {"g": "F", "h": "R", "k": "L"}.get(o) or self._r(["L", "R"])
                    else:
                        a = self._r(["F", "L", "R"], [0.5, 0.25, 0.25])
                if s == "κ" and self.scan_left[k] <= 0 and a != "K":
                    pass
                if a is None:
                    a = self._crw()
            st_out[i] = SI[s]
            a_out[i] = S[a]
            # 掃視結束後的下一個狀態
            if s == "κ" and self.scan_left[k] <= 0:
                self.state[k] = self.after_scan[k] if (self.after_scan[k] != "ν" or w.has_mem[k]) else "ε"
            elif s == "δ":
                self.state[k] = "ε"
            else:
                self.state[k] = s
        return st_out, a_out

    def decide(self, ids, contexts: List[List[int]]) -> np.ndarray:
        raise RuntimeError("EthoTeacher 由 System2 直接以 decide_full 呼叫")

    def learn(self, ids, buffers) -> Dict:
        return {}

    def n_params(self) -> int:
        return 0
