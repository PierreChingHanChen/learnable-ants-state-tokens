"""
System2：Learnable Ant 2 覓食系統
================================
沿用 CwC 的 ForagingSystem（GRAFCET 控制器、fork／resume、事件、介質），只替換：
  A1 初始化：World2、EthoTeacher 或 GPTCore2
  A22 決策：教師產生 (s, a)；GPT 在 dual 模式先說出 s 再決定 a
  回合重設：教師在每趟出巢時重設行為狀態
教師的序列一律以 dual 格式記錄（o s a），bo 資料以 vocab.to_bo 去除狀態詞得到。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..core.field_medium import FieldMedium
from ..foraging.system import ForagingConfig, ForagingSystem
from ..microgpt import MicroGPT
from . import vocab as V
from .core import GPTCore2
from .teacher import EthoTeacher
from .world import MAPS, World2

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "weights")


def base_path(mode: str, tag: str = "") -> str:
    return os.path.join(WEIGHTS_DIR, f"la2_{mode}{tag}.npz")


@dataclass
class LA2Config(ForagingConfig):
    map_name: str = "two_patch"
    core: str = "teacher"                  # 'teacher' / 'gpt'
    mode: str = "dual"                     # GPT：'bo' / 'dual'
    ctx_steps: int = 16                    # 上下文長度（步數）
    n_embd: int = 16
    base_weights: Optional[str] = None     # None → 依 mode 使用 base_path(mode)
    strong_thr: float = 4.0
    give_up: int = 60
    respawn_delay: int = 100
    obs_flip: float = 0.0
    memory: bool = True                    # 食物地點向量開關
    learn_states: bool = True              # dual 線上學習是否也學狀態詞
    teacher_kw: Dict = field(default_factory=dict)
    # ---- 線上學習取樣、教師複習與地圖參數
    weighting: str = "trip"                # 線上學習取樣：'trip'／'step'／'sigma'
    sig_frac: float = 0.5                  # 'sigma' 取樣中強制含 σ 窗口的比例
    rehearse: float = 0.0                  # 每批中教師回合窗口的比例
    rehearse_pool: Optional[list] = None   # 教師回合（dual 格式）
    map_kw: Dict = field(default_factory=dict)   # 地圖參數（例如 per_cell）


class LA2System(ForagingSystem):
    def a1_initialize(self, _m) -> None:
        c = self.cfg
        spec = MAPS[c.map_name](**c.map_kw)
        H, W = spec.shape
        self.medium = FieldMedium(H, W, c.n_ants, rho=c.rho, mode=c.medium_mode, sense_noise=c.sense_noise,
                                  track_provenance=c.track_provenance, rng=self.rng)
        self.world = World2(spec, c.n_ants, self.medium, self.rng, c.trail_threshold, c.mark_amount,
                            strong_thr=c.strong_thr, give_up=c.give_up, respawn_delay=c.respawn_delay,
                            obs_flip=c.obs_flip, memory=c.memory)
        if c.core == "teacher":
            self.core = EthoTeacher(c.n_ants, self.rng, self.world, **c.teacher_kw)
            self.seq_mode = "dual"
        else:
            path = c.base_weights if c.base_weights is not None else base_path(c.mode)
            base = MicroGPT.load(path) if path and os.path.exists(path) else None
            self.core = GPTCore2(c.n_ants, self.rng, c.mode, base, c.ctx_steps, c.weight_mode, c.temperature,
                                 c.lr, c.batch, c.n_updates, c.social, seed=c.seed or 0, n_embd=c.n_embd,
                                 learn_states=c.learn_states, weighting=c.weighting, sig_frac=c.sig_frac,
                                 rehearse=c.rehearse, rehearse_pool=c.rehearse_pool)
            self.seq_mode = c.mode
        A = c.n_ants
        self.alive = np.ones(A, bool)
        self.ctx: List[List[int]] = [[V.BOS_ID] for _ in range(A)]
        self.ep_len = np.zeros(A, int)
        self.ep_marks = np.zeros(A, int)
        self.flags = {"took": np.zeros(A, bool), "delivered": np.zeros(A, bool)}
        self.buffers: Dict[int, List[List[int]]] = {k: [] for k in range(A)}
        self.episodes: List[List[int]] = []
        self.steps = 0
        self.epoch_end = False
        self.finished = False
        self.last_loss = float("nan")
        self.counts = {"mark_carry": 0, "move_carry": 0, "mark_search": 0, "move_search": 0}
        self.state_counts = np.zeros(len(V.STATES), int)
        self.hist: Dict[str, list] = {k: [] for k in (
            "step", "delivered", "rate", "mark_rate_carry", "mark_rate_search", "gate_short", "gate_long",
            "n_alive", "medium_cut", "loss", "success_eps", "timeouts", "takes_rich", "takes_poor",
            "state_counts", "n_mem")}
        self._last_delivered = 0
        self._success = 0
        self._timeouts = 0
        self.probes: List[np.ndarray] = []
        self.g0.mark_done("X0")

    def a22_decide(self, _m) -> None:
        ids = self.ids
        if self.cfg.core == "teacher":
            obs = np.array([self.ctx[k][-1] for k in ids])
            st, self.acts = self.core.decide_full(ids, obs)
            for i, k in enumerate(ids):
                self.ctx[k].append(int(st[i]))
            np.add.at(self.state_counts, [V.STATE_INDEX[int(s)] for s in st], 1)
        else:
            if self.seq_mode == "aux":                 # 說出狀態（記錄用），但不放回模型的輸入
                st, self.acts = self.core.sample_aux(ids, [self.ctx[k] for k in ids])
                for i, k in enumerate(ids):
                    self.ctx[k].append(int(st[i]))
                np.add.at(self.state_counts, [V.STATE_INDEX[int(s)] for s in st], 1)
                self.g0.mark_done("X2")
                return
            if self.seq_mode == "dual":
                st = self.core.sample(ids, [self.ctx[k] for k in ids], V.STATE_MASK)
                for i, k in enumerate(ids):
                    self.ctx[k].append(int(st[i]))
                np.add.at(self.state_counts, [V.STATE_INDEX[int(s)] for s in st], 1)
            self.acts = self.core.sample(ids, [self.ctx[k] for k in ids], V.ACTION_MASK)
        self.g0.mark_done("X2")

    def _reset_ep(self, k) -> None:
        super()._reset_ep(k)
        if self.cfg.core == "teacher":
            self.core.new_trip(int(k))

    def a5_record(self, m) -> None:
        h, w = self.hist, self.world
        h["takes_rich"].append(w.takes[1])
        h["takes_poor"].append(w.takes[0])
        h["state_counts"].append(self.state_counts.copy())
        h["n_mem"].append(int(w.has_mem.sum()))
        self.state_counts[:] = 0
        super().a5_record(m)


def run_la2(**kw):
    return LA2System(LA2Config(**kw)).run()
