"""
覓食世界（A1 地圖、A21 感知編碼、A23 行動執行）
===============================================
* 網格、8 方向朝向（順時針：E, SE, S, SW, W, NW, N, NE），右轉 = +1，左轉 = −1。
* 感知只看前方三格（左前、正前、右前）：螞蟻沒有全域視野。
* 巢方向以「路徑積分」提供（真實螞蟻具備的能力），量化為 前／右／後／左。
* 無效行動為 no-op，不拋出例外（介面契約）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from . import vocab as V

DIRS = np.array([(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)])


@dataclass
class MapSpec:
    name: str
    walls: np.ndarray                  # (H,W) bool
    nest: np.ndarray                   # (H,W) bool
    nest_center: tuple
    food: np.ndarray                   # (H,W) int
    gates: Dict[str, np.ndarray] = field(default_factory=dict)   # 名稱 → (H,W) bool

    @property
    def shape(self):
        return self.walls.shape


def open_field(H: int = 32, W: int = 32, food_per_cell: int = 200) -> MapSpec:
    walls = np.zeros((H, W), bool)
    walls[0, :] = walls[-1, :] = walls[:, 0] = walls[:, -1] = True
    nest = np.zeros((H, W), bool)
    cr, cc = H // 2, W // 2
    nest[cr - 1:cr + 2, cc - 1:cc + 2] = True
    food = np.zeros((H, W), int)
    food[4:7, 4:7] = food_per_cell                    # 左上
    food[H - 7:H - 4, W - 7:W - 4] = food_per_cell    # 右下
    return MapSpec("open_field", walls, nest, (cr, cc), food)


def double_bridge(food_per_cell: int = 200) -> MapSpec:
    """雙橋：巢室（左）與食物室（右）以兩條走廊相連。
    短橋：直線走廊（約 26 格）；長橋：向上繞行（約 45 格）。"""
    H, W = 24, 40
    walls = np.ones((H, W), bool)

    def carve(r0, r1, c0, c1):
        walls[r0:r1 + 1, c0:c1 + 1] = False
    carve(9, 15, 1, 6)          # 巢室
    carve(9, 15, 33, 38)        # 食物室
    carve(11, 13, 7, 32)        # 短橋
    carve(2, 8, 3, 5)           # 長橋：上行
    carve(2, 4, 3, 36)          # 長橋：橫越
    carve(5, 8, 34, 36)         # 長橋：下行
    nest = np.zeros((H, W), bool)
    nest[11:14, 2:5] = True
    food = np.zeros((H, W), int)
    food[11:14, 35:38] = food_per_cell
    gs = np.zeros((H, W), bool); gs[11:14, 20] = True
    gl = np.zeros((H, W), bool); gl[2:5, 20] = True
    return MapSpec("double_bridge", walls, nest, (12, 3), food, {"short": gs, "long": gl})


@dataclass
class ActResult:
    moved: np.ndarray
    blocked: np.ndarray
    took: np.ndarray
    delivered: np.ndarray
    marked: np.ndarray


class World:
    def __init__(self, spec: MapSpec, n_ants: int, medium, rng: np.random.Generator,
                 trail_threshold: float = 0.05, mark_amount: float = 1.0):
        self.spec = spec
        self.walls = spec.walls.copy()
        self.nest = spec.nest
        self.food = spec.food.copy()
        self.food0 = int(self.food.sum())
        self.A = n_ants
        self.medium = medium
        self.rng = rng
        self.trail_threshold = trail_threshold
        self.mark_amount = mark_amount
        nest_cells = np.argwhere(self.nest)
        self.pos = nest_cells[rng.integers(0, len(nest_cells), n_ants)]
        self.heading = rng.integers(0, 8, n_ants)
        self.carrying = np.zeros(n_ants, bool)
        self.delivered_total = 0
        self.delivered_by = np.zeros(n_ants, int)
        self.gate_counts = {g: 0 for g in spec.gates}
        self._in_gate = {g: np.zeros(n_ants, bool) for g in spec.gates}

    # ------------------------------------------------------------ helpers
    def _blocked(self, cells: np.ndarray) -> np.ndarray:
        H, W = self.walls.shape
        r, c = cells[..., 0], cells[..., 1]
        out = (r < 0) | (r >= H) | (c < 0) | (c >= W)
        rr, cc = np.clip(r, 0, H - 1), np.clip(c, 0, W - 1)
        return out | self.walls[rr, cc]

    def cone(self, ids: np.ndarray) -> np.ndarray:
        """(n, 3, 2)：[左前, 正前, 右前]"""
        h = self.heading[ids]
        offs = np.stack([DIRS[(h - 1) % 8], DIRS[h], DIRS[(h + 1) % 8]], axis=1)
        return self.pos[ids][:, None, :] + offs

    def food_total(self) -> int:
        return int(self.food.sum() + self.carrying.sum() + self.delivered_total)

    # ----------------------------------------------------------- A21 感知
    def perceive(self, ids: np.ndarray) -> np.ndarray:
        ids = np.asarray(ids)
        out = np.empty(len(ids), dtype=int)
        if len(ids) == 0:
            return out
        H, W = self.walls.shape
        cone = self.cone(ids)
        blocked = self._blocked(cone)
        cl = np.stack([np.clip(cone[..., 0], 0, H - 1), np.clip(cone[..., 1], 0, W - 1)], -1)
        food_c = (self.food[cl[..., 0], cl[..., 1]] > 0) & ~blocked
        trail = np.where(blocked, 0.0, self.medium.sense(ids, cl))
        p = self.pos[ids]
        here_food = self.food[p[:, 0], p[:, 1]] > 0
        at_nest = self.nest[p[:, 0], p[:, 1]]
        # 巢方向（路徑積分）
        nr, nc = self.spec.nest_center
        vec_ang = np.arctan2(nr - p[:, 0], nc - p[:, 1])
        hd = DIRS[self.heading[ids]]
        h_ang = np.arctan2(hd[:, 0], hd[:, 1])
        rel = (vec_ang - h_ang + np.pi) % (2 * np.pi) - np.pi
        for i, k in enumerate(ids):
            if self.carrying[k]:
                if at_nest[i]:
                    t = "0"
                elif blocked[i, 1]:
                    t = "w"
                elif abs(rel[i]) <= np.pi / 4 + 1e-9:
                    t = "1"
                elif np.pi / 4 < rel[i] <= 3 * np.pi / 4:
                    t = "2"
                elif -3 * np.pi / 4 <= rel[i] < -np.pi / 4:
                    t = "4"
                else:
                    t = "3"
            else:
                if here_food[i]:
                    t = "X"
                elif blocked[i, 1]:
                    t = "W"
                elif food_c[i, 1]:
                    t = "A"
                elif food_c[i, 0]:
                    t = "B"
                elif food_c[i, 2]:
                    t = "C"
                elif trail[i].max() >= self.trail_threshold:
                    j = int(np.argmax(trail[i][[1, 0, 2]]))      # 平手時優先正前
                    t = ["P", "Q", "S"][j]
                else:
                    t = "N"
            out[i] = V.STOI[t]
        return out

    # ----------------------------------------------------------- A23 行動
    def apply(self, ids: np.ndarray, acts: np.ndarray) -> ActResult:
        ids = np.asarray(ids)
        n = len(ids)
        res = ActResult(*(np.zeros(n, bool) for _ in range(5)))
        mark_ids, mark_cells = [], []
        for i, k in enumerate(ids):                     # 依 index 順序處理（仲裁規則）
            a = V.ITOS[int(acts[i])]
            if not V.ACTION_MASK[int(acts[i])]:
                raise ValueError(f"非行動詞 {a!r} 被送入 World.apply（違反序列契約）")
            if a in "FLRflr":
                if a in "Ll":
                    self.heading[k] = (self.heading[k] - 1) % 8
                elif a in "Rr":
                    self.heading[k] = (self.heading[k] + 1) % 8
                tgt = self.pos[k] + DIRS[self.heading[k]]
                if self._blocked(tgt[None])[0]:
                    res.blocked[i] = True
                else:
                    self.pos[k] = tgt
                    res.moved[i] = True
                if a in "flr":
                    res.marked[i] = True
                    mark_ids.append(k)
                    mark_cells.append(self.pos[k].copy())
            elif a == "T":
                r, c = self.pos[k]
                if (not self.carrying[k]) and self.food[r, c] > 0:
                    self.food[r, c] -= 1
                    self.carrying[k] = True
                    self.heading[k] = (self.heading[k] + 4) % 8
                    res.took[i] = True
            elif a == "D":
                r, c = self.pos[k]
                if self.carrying[k] and self.nest[r, c]:
                    self.carrying[k] = False
                    self.delivered_total += 1
                    self.delivered_by[k] += 1
                    self.heading[k] = (self.heading[k] + 4) % 8
                    res.delivered[i] = True
        if mark_ids:
            self.medium.deposit(np.array(mark_ids), np.array(mark_cells), self.mark_amount)
        # 走廊通過計數（進入閘門格的次數）
        for g, gm in self.spec.gates.items():
            inside = gm[self.pos[:, 0], self.pos[:, 1]]
            self.gate_counts[g] += int((inside & ~self._in_gate[g]).sum())
            self._in_gate[g] = inside
        return res

    # ----------------------------------------------------------- 擾動
    def set_walls(self, mask: np.ndarray, value: bool) -> None:
        self.walls[mask] = value
