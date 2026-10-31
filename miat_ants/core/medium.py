"""
Stigmergic Medium（共享介質）
============================

把「環境作為共享記憶」抽象成一個獨立模組（IDEF0 的 Mechanism／共享資料），
讓 agent 只能透過 deposit()／sense() 與彼此間接溝通。這是主題一的關鍵介面：
日後可以把螞蟻換成 LLM 代理人，只要它們遵守同一個介面契約。

介面契約
--------
deposit(agent, edges, amount)  : 寫入（暫存於 delta buffer，週期結束才生效）
commit()                       : 依式(3) τ ← ρ·τ + Δτ 生效（同步更新）
sense(agent) -> (n, n) ndarray : agent 讀到的介質狀態（可能有延遲／雜訊／被切斷）
snapshot() -> (n, n) ndarray   : 觀察者（研究者）看到的真實介質狀態

研究用旋鈕
----------
mode = 'shared'  : 所有 agent 共用一個介質（We sense）
mode = 'private' : 每個 agent 各自一份介質（I think）——同一演算法，只差在
                   介質是否共享，用來量化 stigmergy 的集體增益。
sense_delay      : 讀到 d 個週期前的介質（原論文 5.1.1 感知一致性問題）
sense_noise      : 乘法 log-normal 雜訊 σ
cut()/restore()  : 切斷／恢復介質（原論文 4.2.3「溶解集體智能」的可逆性）
provenance       : 追蹤每一份費洛蒙由哪個 agent 留下（主題二責任歸因）
"""
from __future__ import annotations

from collections import deque
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np


class GraphMedium:
    def __init__(self, n_nodes: int, n_agents: int, tau0: float, rho: float,
                 mode: str = "shared", track_provenance: bool = True,
                 sense_delay: int = 0, sense_noise: float = 0.0,
                 rng: Optional[np.random.Generator] = None):
        if mode not in ("shared", "private"):
            raise ValueError("mode must be 'shared' or 'private'")
        if not (0.0 <= rho <= 1.0):
            raise ValueError("rho（殘留係數）必須介於 0 與 1")
        self.n = n_nodes
        self.A = n_agents
        self.tau0 = float(tau0)
        self.rho = float(rho)
        self.mode = mode
        self.track = track_provenance and mode == "shared"
        self.sense_delay = int(sense_delay)
        self.sense_noise = float(sense_noise)
        self.rng = rng or np.random.default_rng()
        self.is_cut = False
        self.reset()

    # ----------------------------------------------------------------- state
    def reset(self) -> None:
        shape = (self.n, self.n) if self.mode == "shared" else (self.A, self.n, self.n)
        self.tau = np.full(shape, self.tau0, dtype=float)
        self._delta = np.zeros(shape, dtype=float)
        if self.track:
            self.base = np.full((self.n, self.n), self.tau0)          # 環境初始量的殘留
            self.contrib = np.zeros((self.A, self.n, self.n))         # 各 agent 的殘留貢獻
            self._delta_k = np.zeros((self.A, self.n, self.n))
        self._hist: deque = deque(maxlen=max(1, self.sense_delay + 1))
        self._hist.append(self.tau.copy())
        self.commits = 0

    # ------------------------------------------------------------- interface
    def deposit(self, agent: int, edges: Sequence[Tuple[int, int]], amount: float) -> None:
        if not edges:
            return
        e = np.asarray(edges)
        i, j = e[:, 0], e[:, 1]
        if self.mode == "shared":
            np.add.at(self._delta, (i, j), amount)
            np.add.at(self._delta, (j, i), amount)       # 對稱 TSP
            if self.track:
                np.add.at(self._delta_k[agent], (i, j), amount)
                np.add.at(self._delta_k[agent], (j, i), amount)
        else:
            np.add.at(self._delta[agent], (i, j), amount)
            np.add.at(self._delta[agent], (j, i), amount)

    def imprint(self, edges: Sequence[Tuple[int, int]], amount: float) -> None:
        """在初始化時把一條路徑「預先寫入」介質（暖啟動：系統已收斂在該路徑上）。
        這份費洛蒙歸屬於環境（base），不屬於任何 agent。"""
        e = np.asarray(edges)
        i, j = e[:, 0], e[:, 1]
        if self.mode == "shared":
            self.tau[i, j] += amount
            self.tau[j, i] += amount
            if self.track:
                self.base[i, j] += amount
                self.base[j, i] += amount
        else:
            self.tau[:, i, j] += amount
            self.tau[:, j, i] += amount
        self._hist.clear()
        self._hist.append(self.tau.copy())

    def commit(self) -> None:
        """式(3)：τ(t+n) = ρ·τ(t) + Δτ；Δτ 依式(4)(5) 已由 deposit 累加。"""
        self.tau = self.rho * self.tau + self._delta
        self._delta[:] = 0.0
        if self.track:
            self.base *= self.rho
            self.contrib = self.rho * self.contrib + self._delta_k
            self._delta_k[:] = 0.0
        self._hist.append(self.tau.copy())
        self.commits += 1

    def sense(self, agent: int) -> np.ndarray:
        if self.is_cut:
            return np.ones((self.n, self.n))
        src = self._hist[0] if self.sense_delay > 0 else self.tau
        m = src if self.mode == "shared" else src[agent]
        if self.sense_noise > 0:
            m = m * self.rng.lognormal(0.0, self.sense_noise, size=m.shape)
        return m

    def snapshot(self) -> np.ndarray:
        return self.tau if self.mode == "shared" else self.tau.mean(axis=0)

    # ------------------------------------------------------------ disturbance
    def cut(self) -> None:
        self.is_cut = True

    def restore(self) -> None:
        self.is_cut = False

    # -------------------------------------------------------------- provenance
    def provenance_share(self, edges: Iterable[Tuple[int, int]]) -> np.ndarray:
        """回傳長度 A+1 的向量：[環境初始量, agent0, agent1, ...] 在指定邊上的費洛蒙占比。"""
        if not self.track:
            raise RuntimeError("provenance 只在 shared 模式且 track_provenance=True 時可用")
        e = np.asarray(list(edges))
        i, j = e[:, 0], e[:, 1]
        base = self.base[i, j].sum()
        per = self.contrib[:, i, j].sum(axis=1)
        tot = base + per.sum()
        return np.concatenate([[base], per]) / tot

    def check_invariant(self, atol: float = 1e-6) -> bool:
        """τ == base + Σ_k contrib_k（provenance 守恆，單元測試用）。"""
        if not self.track:
            return True
        return bool(np.allclose(self.tau, self.base + self.contrib.sum(axis=0), atol=atol))
