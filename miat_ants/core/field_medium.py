"""
FieldMedium：網格上的費洛蒙場（GraphMedium 的網格版，契約相同）
================================================================
deposit(ids, cells, amount) : 只寫入 delta buffer
commit()                    : τ ← ρ·τ + Δτ
sense(ids, cells)           : 回傳讀值 (n, k)；cut 時回傳 0；private 模式只讀自己的場
snapshot()                  : 研究者視角
"""
from __future__ import annotations

from typing import Optional

import numpy as np


class FieldMedium:
    def __init__(self, H: int, W: int, n_agents: int, rho: float = 0.95, mode: str = "shared",
                 sense_noise: float = 0.0, track_provenance: bool = False,
                 rng: Optional[np.random.Generator] = None):
        if mode not in ("shared", "private"):
            raise ValueError("mode must be 'shared' or 'private'")
        self.H, self.W, self.A = H, W, n_agents
        self.rho = float(rho)
        self.mode = mode
        self.sense_noise = float(sense_noise)
        self.track = track_provenance and mode == "shared"
        self.rng = rng or np.random.default_rng()
        self.is_cut = False
        shape = (H, W) if mode == "shared" else (n_agents, H, W)
        self.tau = np.zeros(shape)
        self._delta = np.zeros(shape)
        if self.track:
            self.contrib = np.zeros((n_agents, H, W))
            self._delta_k = np.zeros((n_agents, H, W))

    def deposit(self, ids: np.ndarray, cells: np.ndarray, amount: float) -> None:
        ids = np.asarray(ids)
        if len(ids) == 0:
            return
        r, c = cells[:, 0], cells[:, 1]
        if self.mode == "shared":
            np.add.at(self._delta, (r, c), amount)
            if self.track:
                np.add.at(self._delta_k, (ids, r, c), amount)
        else:
            np.add.at(self._delta, (ids, r, c), amount)

    def commit(self) -> None:
        self.tau = self.rho * self.tau + self._delta
        self._delta[:] = 0
        if self.track:
            self.contrib = self.rho * self.contrib + self._delta_k
            self._delta_k[:] = 0

    def sense(self, ids: np.ndarray, cells: np.ndarray) -> np.ndarray:
        """cells (n, k, 2) → (n, k)；越界格請由呼叫端先裁切。"""
        if self.is_cut:
            return np.zeros(cells.shape[:2])
        r, c = cells[..., 0], cells[..., 1]
        if self.mode == "shared":
            v = self.tau[r, c]
        else:
            v = self.tau[np.asarray(ids)[:, None], r, c]
        if self.sense_noise > 0:
            v = v * self.rng.lognormal(0, self.sense_noise, v.shape)
        return v

    def snapshot(self) -> np.ndarray:
        return self.tau if self.mode == "shared" else self.tau.mean(0)

    def cut(self) -> None:
        self.is_cut = True

    def restore(self) -> None:
        self.is_cut = False
