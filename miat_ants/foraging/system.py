"""
A0 microGPT 螞蟻覓食系統（MIAT 2.0）
==================================
G0（感覺運動時脈，n=1）：
  X0 init → X1 感知 → X2 決策 → X3 行動 → X4 介質更新
     X4 --¬epoch_end--> X1 ； X4 --epoch_end--> X5 學習 → X6 量測/事件
     X6 --¬finished--> X1 ； X6 --finished--> X7(terminal)
G2（任務意向，族群控制器 n=螞蟻數）：
  X20 搜尋 --took--> X21 返巢 --delivered--> X22 成功回合 --recorded--> X20
  X20/X21 --ep_len≥max_ep--> X23 逾時（例外路徑）--reset_done--> X20
設計說明見 learnable-ants-cwc 儲存庫的 docs/DESIGN_microgpt_ants.md
"""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..core.events import Event, EventLog, EventQueue, PRIORITY_MISSION
from ..core.field_medium import FieldMedium
from ..core.grafcet import Grafcet, SuperLoop
from ..microgpt import MicroGPT
from . import vocab as V
from .cores import GPTCore, RuleCore
from .world import World, MapSpec, open_field, double_bridge

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "weights")
BASE_WEIGHTS = os.path.join(WEIGHTS_DIR, "base_bc.npz")


@dataclass
class ForagingConfig:
    map_name: str = "open_field"           # 'open_field' / 'double_bridge'
    n_ants: int = 30
    core: str = "rule"                     # 'rule' / 'gpt'
    weight_mode: str = "individual"        # GPT：'individual'（共享基礎＋各自微調）/ 'shared'
    base_weights: Optional[str] = BASE_WEIGHTS   # None = 隨機初始化
    learn: bool = True                     # GPT：線上篩選式自我模仿
    social: bool = False                   # GPT：也從他者的成功回合學習（直接模仿）
    temperature: float = 1.0
    lr: float = 3e-3
    batch: int = 16
    n_updates: int = 4
    buffer_size: int = 20                  # 每隻螞蟻保留最近幾個成功回合
    max_steps: int = 3000
    steps_per_epoch: int = 50
    max_ep_steps: int = 400
    rho: float = 0.97                      # 費洛蒙殘留係數（每步）
    trail_threshold: float = 0.05
    mark_amount: float = 1.0
    medium_mode: str = "shared"            # 'shared' / 'private'
    sense_noise: float = 0.0
    seed: Optional[int] = None
    perturbations: Sequence[Event] = field(default_factory=list)
    collect_episodes: bool = False         # 收集所有回合（行為複製資料）
    record_trace: bool = False
    probe_contexts: Optional[np.ndarray] = None   # 每個 epoch 量測的探針上下文
    track_provenance: bool = False         # 介質來源追蹤（主題二：誰留下了路徑）
    rule_mark_p: float = 1.0               # 規則螞蟻攜帶時的標記機率（exp11b：讓標記行為可被學習改變）


@dataclass
class ForagingResult:
    history: Dict[str, list]
    delivered: int
    events: EventLog
    world: World
    core: object
    cfg: ForagingConfig
    episodes: List[List[int]]
    probes: List[np.ndarray]


class ForagingSystem:
    def __init__(self, cfg: ForagingConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.log = EventLog()
        self.queue = EventQueue(cfg.perturbations)
        self._build_model()

    # ================================================================ MODEL
    def _build_model(self) -> None:
        A = self.cfg.n_ants
        g0 = Grafcet("G0", idef0="A0", event_log=self.log)
        g2 = Grafcet("G2", n=A, idef0="A2", event_log=self.log)
        self.g0, self.g2 = g0, g2
        d0, d2 = g0.done, g2.done
        g0.add_step("X0", self.a1_initialize, "P1", initial=True, desc="初始化", idef0="A1")
        g0.add_step("X1", self.a21_perceive, "P1", desc="感知編碼", idef0="A21")
        g0.add_step("X2", self.a22_decide, "P1", desc="決策（核心）", idef0="A22")
        g0.add_step("X3", self.a23_act, "P1", desc="行動執行", idef0="A23")
        g0.add_step("X4", self.a3_medium, "P1", desc="介質更新", idef0="A3")
        g0.add_step("X5", self.a4_learn, "P1", desc="線上學習", idef0="A4")
        g0.add_step("X6", self.a5_record, "P1", desc="量測／事件", idef0="A5")
        g0.add_step("X7", self.a6_report, "P1", terminal=True, desc="輸出結果", idef0="A5")
        g0.add_transition("T0", ["X0"], ["X1"], lambda: d0["X0"], "init_done")
        g0.add_transition("T1", ["X1"], ["X2"], lambda: d0["X1"], "obs_ready")
        g0.add_transition("T2", ["X2"], ["X3"], lambda: d0["X2"], "actions_ready")
        g0.add_transition("T3", ["X3"], ["X4"], lambda: d0["X3"], "act_done")
        g0.add_transition("T4a", ["X4"], ["X1"], lambda: d0["X4"] & (not self.epoch_end), "step_done ∧ ¬epoch_end")
        g0.add_transition("T4b", ["X4"], ["X5"], lambda: d0["X4"] & self.epoch_end, "step_done ∧ epoch_end")
        g0.add_transition("T5", ["X5"], ["X6"], lambda: d0["X5"], "learn_done")
        g0.add_transition("T6a", ["X6"], ["X1"], lambda: d0["X6"] & (not self.finished), "record_done ∧ ¬finished")
        g0.add_transition("T6b", ["X6"], ["X7"], lambda: d0["X6"] & self.finished, "record_done ∧ finished")

        g2.add_step("X20", None, initial=True, desc="搜尋", idef0="A2")
        g2.add_step("X21", None, desc="攜帶返巢", idef0="A2")
        g2.add_step("X22", self.a24_record_success, "P1", desc="成功回合：寫入緩衝", idef0="A4")
        g2.add_step("X23", self.a25_reset_timeout, "P1", desc="逾時：捨棄回合", idef0="A2")
        g2.add_transition("T20t", ["X20"], ["X23"], lambda: self._timeout(), "ep_len ≥ max_ep（例外）")
        g2.add_transition("T20", ["X20"], ["X21"], lambda: self._flag("took"), "took")
        g2.add_transition("T21t", ["X21"], ["X23"], lambda: self._timeout(), "ep_len ≥ max_ep（例外）")
        g2.add_transition("T21", ["X21"], ["X22"], lambda: self._flag("delivered"), "delivered")
        g2.add_transition("T22", ["X22"], ["X20"], lambda: d2["X22"], "recorded")
        g2.add_transition("T23", ["X23"], ["X20"], lambda: d2["X23"], "reset_done")
        for g in (g0, g2):
            g.record_trace = self.cfg.record_trace
            issues = g.validate()
            if issues:
                raise ValueError(f"{g.name} 模型審查未通過: {issues}")
        self.loop = SuperLoop([g0, g2])

    # ============================================================ CONDITIONS
    def _flag(self, name):
        if not hasattr(self, "flags"):
            return False
        return self.flags[name]

    def _timeout(self):
        if not hasattr(self, "ep_len"):
            return False
        return self.ep_len >= self.cfg.max_ep_steps

    # =============================================================== ACTIONS
    def a1_initialize(self, _m) -> None:
        c = self.cfg
        spec: MapSpec = {"open_field": open_field, "double_bridge": double_bridge}[c.map_name]()
        H, W = spec.shape
        self.medium = FieldMedium(H, W, c.n_ants, rho=c.rho, mode=c.medium_mode,
                                  sense_noise=c.sense_noise, track_provenance=c.track_provenance,
                                  rng=self.rng)
        self.world = World(spec, c.n_ants, self.medium, self.rng, c.trail_threshold, c.mark_amount)
        if c.core == "rule":
            self.core = RuleCore(c.n_ants, self.rng, mark_p=c.rule_mark_p)
        else:
            base = MicroGPT.load(c.base_weights) if c.base_weights else None
            self.core = GPTCore(c.n_ants, self.rng, base, c.weight_mode, c.temperature, c.lr,
                                c.batch, c.n_updates, c.social, seed=c.seed or 0)
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
        self.hist: Dict[str, list] = {k: [] for k in (
            "step", "delivered", "rate", "mark_rate_carry", "mark_rate_search", "gate_short",
            "gate_long", "n_alive", "medium_cut", "loss", "success_eps", "timeouts")}
        if self.medium.track:
            for k in ("contrib_trail", "contrib_total", "delivered_by"):
                self.hist[k] = []
        self._last_delivered = 0
        self._success = 0
        self._timeouts = 0
        self.probes: List[np.ndarray] = []
        self.g0.mark_done("X0")

    def a21_perceive(self, _m) -> None:
        for k in self.flags:
            self.flags[k][:] = False
        self.ids = np.flatnonzero(self.alive)
        obs = self.world.perceive(self.ids)
        for k, o in zip(self.ids, obs):
            self.ctx[k].append(int(o))
        self.g0.mark_done("X1")

    def a22_decide(self, _m) -> None:
        contexts = [self.ctx[k][-V_BLOCK:] for k in self.ids]
        self.acts = self.core.decide(self.ids, contexts)
        self.g0.mark_done("X2")

    def a23_act(self, _m) -> None:
        carrying_before = self.world.carrying[self.ids].copy()
        res = self.world.apply(self.ids, self.acts)
        for i, k in enumerate(self.ids):
            self.ctx[k].append(int(self.acts[i]))
        self.ep_len[self.ids] += 1
        mv = res.moved | res.blocked
        self.counts["move_carry"] += int((mv & carrying_before).sum())
        self.counts["mark_carry"] += int((res.marked & carrying_before).sum())
        self.counts["move_search"] += int((mv & ~carrying_before).sum())
        self.counts["mark_search"] += int((res.marked & ~carrying_before).sum())
        self.ep_marks[self.ids] += res.marked
        self.flags["took"][self.ids] = res.took
        self.flags["delivered"][self.ids] = res.delivered
        for i in np.flatnonzero(res.took):
            self.log.emit("take", self.steps, f"ant:{int(self.ids[i])}", {"pos": self.world.pos[self.ids[i]].tolist()})
        for i in np.flatnonzero(res.delivered):
            self.log.emit("deliver", self.steps, f"ant:{int(self.ids[i])}", {"ep_len": int(self.ep_len[self.ids[i]])})
        self.g0.mark_done("X3")

    def a3_medium(self, _m) -> None:
        self.medium.commit()
        self.steps += 1
        self.epoch_end = self.steps % self.cfg.steps_per_epoch == 0 or self.steps >= self.cfg.max_steps
        self.g0.mark_done("X4")

    def a24_record_success(self, m) -> None:
        for k in np.flatnonzero(m):
            seq = self.ctx[k]
            self.buffers[k].append(seq)
            self.buffers[k] = self.buffers[k][-self.cfg.buffer_size:]
            if self.cfg.collect_episodes:
                self.episodes.append(seq)
            self.log.emit("episode_end", self.steps, f"ant:{int(k)}",
                          {"success": True, "length": int(self.ep_len[k]),
                           "mark_rate": float(self.ep_marks[k] / max(self.ep_len[k], 1))})
            self._success += 1
            self._reset_ep(k)
        self.g2.mark_done("X22", m)

    def a25_reset_timeout(self, m) -> None:
        for k in np.flatnonzero(m):
            if self.cfg.collect_episodes:
                self.episodes.append(self.ctx[k])
            self.log.emit("episode_end", self.steps, f"ant:{int(k)}",
                          {"success": False, "length": int(self.ep_len[k])})
            self._timeouts += 1
            self._reset_ep(k)
        self.g2.mark_done("X23", m)

    def _reset_ep(self, k) -> None:
        self.ctx[k] = [V.BOS_ID]
        self.ep_len[k] = 0
        self.ep_marks[k] = 0

    def a4_learn(self, _m) -> None:
        if self.cfg.core == "gpt" and self.cfg.learn:
            out = self.core.learn(np.flatnonzero(self.alive), self.buffers)
            self.last_loss = out.get("loss", float("nan"))
            self.log.emit("learn", self.steps, "A4", out, PRIORITY_MISSION)
        self.g0.mark_done("X5")

    def a5_record(self, _m) -> None:
        h, c, w = self.hist, self.counts, self.world
        h["step"].append(self.steps)
        h["delivered"].append(w.delivered_total)
        h["rate"].append((w.delivered_total - self._last_delivered) / max(1, self.cfg.steps_per_epoch))
        self._last_delivered = w.delivered_total
        h["mark_rate_carry"].append(c["mark_carry"] / max(1, c["move_carry"]))
        h["mark_rate_search"].append(c["mark_search"] / max(1, c["move_search"]))
        for k in c:
            c[k] = 0
        h["gate_short"].append(w.gate_counts.get("short", 0))
        h["gate_long"].append(w.gate_counts.get("long", 0))
        h["n_alive"].append(int(self.alive.sum()))
        h["medium_cut"].append(bool(self.medium.is_cut))
        h["loss"].append(self.last_loss)
        h["success_eps"].append(self._success)
        h["timeouts"].append(self._timeouts)
        if self.medium.track:
            # 痕跡（provenance）：各螞蟻在「可被感知的路徑」上（τ ≥ 門檻）的費洛蒙量，以及全場總量
            trail = self.medium.tau >= self.cfg.trail_threshold
            h["contrib_trail"].append(self.medium.contrib[:, trail].sum(1))
            h["contrib_total"].append(self.medium.contrib.sum((1, 2)))
            h["delivered_by"].append(w.delivered_by.copy())
        if self.cfg.probe_contexts is not None and hasattr(self.core, "policy_probe"):
            self.probes.append(self.core.policy_probe(self.cfg.probe_contexts))
        for ev in self.queue.pop_due(self.steps):
            self._apply_perturbation(ev)
        self.finished = self.steps >= self.cfg.max_steps
        self.g0.mark_done("X6")

    def _apply_perturbation(self, ev: Event) -> None:
        kind, p = ev.event_type, ev.payload
        if kind == "cut_medium":
            self.medium.cut()
        elif kind == "restore_medium":
            self.medium.restore()
        elif kind in ("block_gate", "unblock_gate"):
            self.world.set_walls(self.world.spec.gates[p["gate"]], kind == "block_gate")
        elif kind == "remove_ants":
            if "ids" in p:                     # 指定移除（反事實實驗）
                self.alive[np.asarray(p["ids"], dtype=int)] = False
            else:
                live = np.flatnonzero(self.alive)
                n = int(round(p.get("fraction", 0.5) * len(live)))
                self.alive[self.rng.choice(live, n, replace=False)] = False
        elif kind == "reseed":
            # 研究者事件：以指定種子重設共用亂數流（world、medium、core 共用同一個 Generator 物件）
            self.rng.bit_generator.state = np.random.default_rng(p["seed"]).bit_generator.state
        elif kind == "restore_ants":
            self.alive[:] = True
        else:
            raise ValueError(f"未知的擾動事件 {kind}")
        self.log.emit("perturbation_applied", self.steps, ev.source_module, {"kind": kind, **p}, ev.priority)

    def a6_report(self, _m) -> None:
        self.log.emit("report", self.steps, "A5", {"delivered": int(self.world.delivered_total)})
        self._reported = True

    # ================================================================== RUN
    def run(self) -> ForagingResult:
        self.loop.reset()
        self._reported = False
        self.loop.run(until=lambda: self._reported)
        return self._result()

    def _result(self) -> ForagingResult:
        return ForagingResult(self.hist, int(self.world.delivered_total), self.log, self.world,
                              self.core, self.cfg, self.episodes, self.probes)

    # ======================================================= CHECKPOINT / FORK
    # 研究者工具（不屬於受控系統）：在 epoch 邊界暫停，複製整個 Datapath 狀態與 GRAFCET 標記，
    # 讓同一段歷史分岔成多個反事實分支（配對設計：分岔前的歷史完全相同）。
    _CONTROL_ATTRS = ("cfg", "g0", "g2", "loop", "log", "queue")

    def run_until(self, step: int) -> "ForagingSystem":
        """從頭執行到第 step 步的 epoch 邊界（X6 完成）後暫停。step 必須是 steps_per_epoch 的倍數。"""
        if step % self.cfg.steps_per_epoch:
            raise ValueError("暫停點必須落在 epoch 邊界")
        self.loop.reset()
        self._reported = False
        self.loop.run(until=lambda: getattr(self, "steps", -1) >= step
                      and bool(self.g0.X["X6"].any() and self.g0.done["X6"].any()))
        return self

    def fork(self, events_now: Sequence[Event] = (), perturbations: Sequence[Event] = ()) -> "ForagingSystem":
        """複製目前狀態成一個新系統；events_now 立即套用，perturbations 為之後的排程事件。"""
        new = ForagingSystem.__new__(ForagingSystem)
        new.cfg = replace(self.cfg, perturbations=list(perturbations))
        new.log = EventLog()
        new.queue = EventQueue(new.cfg.perturbations)
        state = {k: v for k, v in self.__dict__.items() if k not in self._CONTROL_ATTRS}
        new.__dict__.update(copy.deepcopy(state))
        new._build_model()
        for g_new, g_old in ((new.g0, self.g0), (new.g2, self.g2)):
            for attr in ("X", "done", "_entered"):       # 就地更新：轉移條件以閉包持有這些 dict
                d_new, d_old = getattr(g_new, attr), getattr(g_old, attr)
                d_new.clear()
                d_new.update({k: v.copy() for k, v in d_old.items()})
            g_new.clock = g_old.clock
        new.loop.clock = self.loop.clock
        for ev in events_now:
            new._apply_perturbation(ev)
        return new

    def resume(self) -> ForagingResult:
        """從暫停點（或 fork 後）繼續執行到 max_steps。"""
        self.loop.run(until=lambda: self._reported)
        return self._result()


V_BLOCK = 16


def run_foraging(**kw) -> ForagingResult:
    return ForagingSystem(ForagingConfig(**kw)).run()
