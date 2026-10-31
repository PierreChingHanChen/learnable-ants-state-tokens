"""
World2：在 CwC 覓食世界上加入文獻行為所需的機制
===============================================
* 食物品質（優／劣，Beckers et al. 1993）：腳下食物以 X／Y 呈現。
* 食物地點向量（Bolek et al. 2012）：拿到食物時記住地點；下一趟出巢時，導航模組以 g/h/j/k/z
  提供目標方向（與 CwC 以 1–4 提供巢方向的做法相同，視為路徑積分的輸出）。
* 放棄規則：到達記憶地點卻沒有食物後，超過 give_up 步即遺忘向量（系統搜尋的時間上限）。
* 攜帶時感知腳下路徑強弱（Czaczkes et al. 2013）：巢方向 1–4（弱）或 5–8（強）。
* 行動 U（Beckers et al. 1992）與 K（Wystrach et al. 2014）。
* 食物塊耗盡後於新位置重生（two_patch 地圖），產生「記憶地點已空」的情境。
* 感知翻轉雜訊 obs_flip：以機率 p 把感知詞換成同組的隨機感知詞（E3）。
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..foraging.world import DIRS, MapSpec, World, ActResult, double_bridge
from . import vocab as V


def two_patch(per_cell: int = 30) -> MapSpec:
    H = W = 32
    walls = np.zeros((H, W), bool)
    walls[0, :] = walls[-1, :] = walls[:, 0] = walls[:, -1] = True
    nest = np.zeros((H, W), bool)
    cr, cc = H // 2, W // 2
    nest[cr - 1:cr + 2, cc - 1:cc + 2] = True
    food = np.zeros((H, W), int)
    food[4:7, 4:7] = per_cell            # 優質
    food[25:28, 25:28] = per_cell        # 劣質
    spec = MapSpec("two_patch", walls, nest, (cr, cc), food)
    q = np.zeros((H, W), int)
    q[4:7, 4:7] = 1
    spec.quality = q
    spec.patches = [dict(r=5, c=5, q=1), dict(r=26, c=26, q=0)]
    spec.per_cell = per_cell
    return spec


def double_bridge2() -> MapSpec:
    spec = double_bridge()
    spec.quality = (spec.food > 0).astype(int)      # 全部優質
    spec.patches = None
    return spec


def barrier(per_cell: int = 30) -> MapSpec:
    """泛化測試用的未見地圖：36×36、巢偏離中心、兩道內牆、三個食物塊（兩優一劣）。"""
    H = W = 36
    walls = np.zeros((H, W), bool)
    walls[0, :] = walls[-1, :] = walls[:, 0] = walls[:, -1] = True
    walls[14, 8:28] = True                 # 巢上方的水平牆
    walls[18:31, 12] = True                # 巢左側的垂直牆
    nest = np.zeros((H, W), bool)
    cr, cc = 24, 24
    nest[cr - 1:cr + 2, cc - 1:cc + 2] = True
    patches = [dict(r=6, c=6, q=1), dict(r=6, c=29, q=1), dict(r=29, c=6, q=0)]
    food = np.zeros((H, W), int)
    q = np.zeros((H, W), int)
    for pt in patches:
        food[pt["r"] - 1:pt["r"] + 2, pt["c"] - 1:pt["c"] + 2] = per_cell
        q[pt["r"] - 1:pt["r"] + 2, pt["c"] - 1:pt["c"] + 2] = pt["q"]
    spec = MapSpec("barrier", walls, nest, (cr, cc), food)
    spec.quality = q
    spec.patches = patches
    spec.per_cell = per_cell
    return spec


MAPS = {"two_patch": two_patch, "double_bridge": double_bridge2, "barrier": barrier}


def _rel_dir(p, target, heading):
    """target 相對朝向的方向：0 前、1 右、2 後、3 左。"""
    vec = np.arctan2(target[..., 0] - p[..., 0], target[..., 1] - p[..., 1])
    hd = DIRS[heading]
    h_ang = np.arctan2(hd[..., 0], hd[..., 1])
    rel = (vec - h_ang + np.pi) % (2 * np.pi) - np.pi
    out = np.full(rel.shape, 2)
    out[np.abs(rel) <= np.pi / 4 + 1e-9] = 0
    out[(rel > np.pi / 4) & (rel <= 3 * np.pi / 4)] = 1
    out[(rel < -np.pi / 4) & (rel >= -3 * np.pi / 4)] = 3
    return out


GROUPS = [np.array([V.STOI[o] for o in g]) for g in (V.OBS_SEARCH, V.OBS_GOAL, V.OBS_CARRY)]


class World2(World):
    def __init__(self, spec: MapSpec, n_ants: int, medium, rng: np.random.Generator,
                 trail_threshold: float = 0.05, mark_amount: float = 1.0, strong_thr: float = 4.0,
                 give_up: int = 60, respawn_delay: int = 100, obs_flip: float = 0.0,
                 memory: bool = True):
        super().__init__(spec, n_ants, medium, rng, trail_threshold, mark_amount)
        self.quality = spec.quality.copy()
        self.strong_thr = strong_thr
        self.give_up = give_up
        self.respawn_delay = respawn_delay
        self.obs_flip = obs_flip
        self.memory = memory                    # False：關閉食物地點向量（對照 Beckers 1993 的設定）
        self.patches = [dict(p) for p in spec.patches] if spec.patches else None
        self.respawn_t: List[int] = [-1] * (len(self.patches) if self.patches else 0)
        self.t = 0
        A = n_ants
        self.has_mem = np.zeros(A, bool)
        self.mem_pos = np.zeros((A, 2), int)
        self.mem_q = np.zeros(A, int)
        self.arrive_t = np.full(A, -1)          # 到達空的記憶地點後經過的步數（-1 = 尚未）
        self.carry_q = np.zeros(A, int)         # 攜帶中的食物品質
        self.takes = {0: 0, 1: 0}               # 依品質累計拿取次數
        self.last_obs = np.full(A, -1)

    # ------------------------------------------------------------ 感知
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
        here = self.medium.sense(ids, p[:, None, :])[:, 0]
        here_food = self.food[p[:, 0], p[:, 1]] > 0
        at_nest = self.nest[p[:, 0], p[:, 1]]
        nest_dir = _rel_dir(p, np.array(self.spec.nest_center)[None], self.heading[ids])
        goal_dir = _rel_dir(p, self.mem_pos[ids], self.heading[ids])
        goal_dist = np.abs(p - self.mem_pos[ids]).max(1)
        for i, k in enumerate(ids):
            if self.carrying[k]:
                if at_nest[i]:
                    t = "0"
                elif blocked[i, 1]:
                    t = "w"
                else:
                    t = str(1 + nest_dir[i] + (4 if here[i] >= self.strong_thr else 0))
            else:
                if here_food[i]:
                    t = "X" if self.quality[p[i, 0], p[i, 1]] == 1 else "Y"
                elif blocked[i, 1]:
                    t = "W"
                elif food_c[i, 1]:
                    t = "A"
                elif food_c[i, 0]:
                    t = "B"
                elif food_c[i, 2]:
                    t = "C"
                elif self.has_mem[k]:
                    t = "z" if goal_dist[i] <= 1 else "ghjk"[goal_dir[i]]
                elif trail[i].max() >= self.trail_threshold:
                    t = ["P", "Q", "S"][int(np.argmax(trail[i][[1, 0, 2]]))]
                else:
                    t = "N"
            tok = V.STOI[t]
            if self.obs_flip > 0 and self.rng.random() < self.obs_flip and t not in "XY0":
                g = next(G for G in GROUPS if tok in G)
                tok = int(g[self.rng.integers(len(g))])
            out[i] = tok
        self.last_obs[ids] = out
        return out

    def goal_dist(self, k: int) -> int:
        return int(np.abs(self.pos[k] - self.mem_pos[k]).max())

    # ------------------------------------------------------------ 行動
    def apply(self, ids: np.ndarray, acts: np.ndarray) -> ActResult:
        ids = np.asarray(ids)
        n = len(ids)
        res = ActResult(*(np.zeros(n, bool) for _ in range(5)))
        mark_ids, mark_cells = [], []
        for i, k in enumerate(ids):
            a = V.ITOS[int(acts[i])]
            if not V.ACTION_MASK[int(acts[i])]:
                raise ValueError(f"非行動詞 {a!r} 被送入 World2.apply")
            if a in "FLRflrU":
                if a in "Ll":
                    self.heading[k] = (self.heading[k] - 1) % 8
                elif a in "Rr":
                    self.heading[k] = (self.heading[k] + 1) % 8
                elif a == "U":
                    self.heading[k] = (self.heading[k] + 4) % 8
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
            elif a == "K":
                self.heading[k] = (self.heading[k] + (2 if self.rng.random() < 0.5 else -2)) % 8
            elif a == "T":
                r, c = self.pos[k]
                if (not self.carrying[k]) and self.food[r, c] > 0:
                    self.food[r, c] -= 1
                    self.carrying[k] = True
                    q = int(self.quality[r, c])
                    self.carry_q[k] = q
                    self.takes[q] += 1
                    self.has_mem[k] = self.memory              # 食物地點向量（Bolek et al. 2012）
                    self.mem_pos[k] = (r, c)
                    self.mem_q[k] = q
                    self.arrive_t[k] = -1
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
        for g, gm in self.spec.gates.items():
            inside = gm[self.pos[:, 0], self.pos[:, 1]]
            self.gate_counts[g] += int((inside & ~self._in_gate[g]).sum())
            self._in_gate[g] = inside
        self._memory_update(ids)
        self._respawn()
        self.t += 1
        return res

    def _memory_update(self, ids) -> None:
        for k in ids:
            if not self.has_mem[k] or self.carrying[k]:
                continue
            r, c = self.pos[k]
            if self.arrive_t[k] < 0:
                if self.goal_dist(k) <= 1 and self.food[max(r - 1, 0):r + 2, max(c - 1, 0):c + 2].sum() == 0:
                    self.arrive_t[k] = 0
            else:
                self.arrive_t[k] += 1
                if self.arrive_t[k] > self.give_up:            # 放棄搜尋，遺忘向量
                    self.has_mem[k] = False
                    self.arrive_t[k] = -1

    def _respawn(self) -> None:
        if not self.patches:
            return
        H, W = self.walls.shape
        for j, pt in enumerate(self.patches):
            r, c = pt["r"], pt["c"]
            if self.respawn_t[j] < 0 and self.food[r - 1:r + 2, c - 1:c + 2].sum() == 0:
                self.respawn_t[j] = self.t + self.respawn_delay
                self.quality[r - 1:r + 2, c - 1:c + 2] = 0
            elif self.respawn_t[j] >= 0 and self.t >= self.respawn_t[j]:
                nr, nc = self.spec.nest_center
                for _ in range(200):                          # 新位置：距巢 9–13 格，與他塊不重疊
                    ang = self.rng.uniform(0, 2 * np.pi)
                    d = self.rng.uniform(9, 13)
                    rr, cc = int(round(nr + d * np.sin(ang))), int(round(nc + d * np.cos(ang)))
                    if 2 <= rr <= H - 3 and 2 <= cc <= W - 3 and \
                            self.food[rr - 2:rr + 3, cc - 2:cc + 3].sum() == 0 and \
                            not self.walls[rr - 1:rr + 2, cc - 1:cc + 2].any() and \
                            not self.nest[rr - 1:rr + 2, cc - 1:cc + 2].any():      # 新位置不可落在牆上或巢內（two-patch 無內牆，此檢查不會生效）
                        break
                self.food[rr - 1:rr + 2, cc - 1:cc + 2] = self.spec.per_cell
                self.quality[rr - 1:rr + 2, cc - 1:cc + 2] = pt["q"]
                pt["r"], pt["c"] = rr, cc
                self.respawn_t[j] = -1

    def food_total(self) -> int:     # 重生使總量不守恆；僅供測試以外的參考
        return int(self.food.sum() + self.carrying.sum() + self.delivered_total)
