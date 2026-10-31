"""
GRAFCET 執行引擎（MIAT 2.0, Ch.3–4）
=====================================

依《MIAT 系統設計方法論》第四章的 Grafcet-to-C 合成規則，以 Python 實作：

    Fundamental Block : if (Xi && Ri) { Xi = 0; Xj = 1; }
    Divergence AND    : if (Xi && Ri) { Xi = 0; Xj = 1; Xk = 1; }
    Convergence AND   : if (Xi && Xj && Ri) { Xi = 0; Xj = 0; Xk = 1; }
    Divergence OR     : if / else-if（宣告順序即優先序）
    Reset             : 初始狀態 = 1，其餘 = 0

設計重點
--------
1. Controller 與 Datapath 分離：`controller()` 只做狀態轉移，
   `action_datapath()` 只執行 active 狀態的 Action。兩者不得混寫。
2. 同步語意：同一個 clock 內，所有 transition 以「clock 開始時的狀態快照」
   判斷是否致能，再一次性更新（對應硬體同步控制器）。
3. 族群向量化（population GRAFCET）：狀態變數 Xi 是長度 n 的布林陣列。
   n = 1 即一般單一控制器；n = 螞蟻數，則相當於「每隻螞蟻一個獨立
   Controller」，以相同合成規則逐元素平行運算（對應硬體上複製的控制器）。
4. 每個轉移都必須有條件；不允許以 True 取代完成事件（Gate 2）。
   Action 若需時間，完成時由 Action 呼叫 `mark_done()` 產生完成事件。
5. 可追溯：轉移觸發可寫入事件契約（EventLog），並可記錄波形 trace。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

Mask = np.ndarray  # bool array, shape (n,)
Condition = Callable[[], "bool | np.ndarray"]
Action = Callable[[Mask], None]


@dataclass
class Step:
    sid: str
    desc: str = ""
    action: Optional[Action] = None
    # 'N'  : 狀態 active 期間每個 clock 都執行（continuous action）
    # 'P1' : 只在進入狀態的那一個 clock 執行一次（impulse action）
    mode: str = "N"
    initial: bool = False
    terminal: bool = False
    idef0: str = ""          # 對應的 IDEF0 模組編號（Traceability）


@dataclass
class Transition:
    tid: str
    src: Tuple[str, ...]
    dst: Tuple[str, ...]
    cond: Condition
    desc: str = ""           # 條件的文字描述（用於模型輸出與審查）


class GrafcetError(Exception):
    pass


class Grafcet:
    """單一 GRAFCET 控制器（或 n 個同構控制器的族群）。"""

    def __init__(self, name: str, n: int = 1, idef0: str = "", event_log=None):
        self.name = name
        self.n = int(n)
        self.idef0 = idef0
        self.steps: Dict[str, Step] = {}
        self.transitions: List[Transition] = []
        self.X: Dict[str, Mask] = {}
        self.done: Dict[str, Mask] = {}
        self._entered: Dict[str, Mask] = {}
        self.clock = 0
        self.event_log = event_log
        self.log_transitions = False      # 族群控制器預設不逐一記錄（量大）
        self.record_trace = False
        self.trace: List[Tuple[int, Tuple[str, ...], Tuple[str, ...]]] = []
        self.conflicts = 0                # Divergence OR 同時致能次數（審查用）

    # ------------------------------------------------------------------ model
    def add_step(self, sid: str, action: Optional[Action] = None, mode: str = "N",
                 initial: bool = False, terminal: bool = False, desc: str = "",
                 idef0: str = "") -> "Grafcet":
        if sid in self.steps:
            raise GrafcetError(f"{self.name}: duplicated step {sid}")
        if mode not in ("N", "P1"):
            raise GrafcetError(f"{self.name}.{sid}: action mode must be 'N' or 'P1'")
        self.steps[sid] = Step(sid, desc, action, mode, initial, terminal, idef0)
        return self

    def add_transition(self, tid: str, src: Sequence[str], dst: Sequence[str],
                       cond: Condition, desc: str = "") -> "Grafcet":
        if cond is None or cond is True or cond is False:
            raise GrafcetError(
                f"{self.name}.{tid}: 每個轉移都必須有可量測的條件函式（不可寫 True）")
        src, dst = tuple(src), tuple(dst)
        for s in src + dst:
            if s not in self.steps:
                raise GrafcetError(f"{self.name}.{tid}: unknown step {s}")
        self.transitions.append(Transition(tid, src, dst, cond, desc))
        return self

    # --------------------------------------------------------------- validate
    def validate(self) -> List[str]:
        """Gate 2 模型審查：回傳問題清單（空清單表示通過）。"""
        issues = []
        inits = [s for s in self.steps.values() if s.initial]
        if not inits:
            issues.append("沒有初始狀態")
        out = {sid: [] for sid in self.steps}
        inc = {sid: [] for sid in self.steps}
        for t in self.transitions:
            for s in t.src:
                out[s].append(t.tid)
            for d in t.dst:
                inc[d].append(t.tid)
        for sid, st in self.steps.items():
            if not st.initial and not inc[sid]:
                issues.append(f"{sid} 不是初始狀態，卻沒有進入條件")
            if not st.terminal and not out[sid]:
                issues.append(f"{sid} 沒有離開條件（若為終止狀態請標記 terminal）")
            if not t_desc_ok(self, sid):
                issues.append(f"{sid} 的離開轉移缺少條件描述")
        # 可達性
        reach = {s.sid for s in inits}
        changed = True
        while changed:
            changed = False
            for t in self.transitions:
                if all(s in reach for s in t.src):
                    for d in t.dst:
                        if d not in reach:
                            reach.add(d)
                            changed = True
        for sid in self.steps:
            if sid not in reach:
                issues.append(f"{sid} 由初始狀態不可達")
        return issues

    # ------------------------------------------------------------------ reset
    def reset(self) -> None:
        for sid, st in self.steps.items():
            self.X[sid] = np.full(self.n, st.initial, dtype=bool)
            self.done[sid] = np.zeros(self.n, dtype=bool)
            self._entered[sid] = np.full(self.n, st.initial, dtype=bool)
        self.clock = 0
        self.trace.clear()
        self.conflicts = 0

    # ------------------------------------------------------------- controller
    def controller(self) -> List[str]:
        """只處理狀態轉移（同步語意，宣告順序 = OR 分支優先序）。"""
        snap = {k: v.copy() for k, v in self.X.items()}
        consumed = {k: np.zeros(self.n, dtype=bool) for k in self.X}
        fired: List[str] = []
        for t in self.transitions:
            en = np.ones(self.n, dtype=bool)
            for s in t.src:
                en &= snap[s]
            if not en.any():
                continue
            c = t.cond()
            c = np.broadcast_to(np.asarray(c, dtype=bool), (self.n,))
            en &= c
            if not en.any():
                continue
            blocked = np.zeros(self.n, dtype=bool)
            for s in t.src:
                blocked |= consumed[s]
            if (en & blocked).any():
                self.conflicts += int((en & blocked).sum())
            en &= ~blocked
            if not en.any():
                continue
            for s in t.src:
                self.X[s] &= ~en
                consumed[s] |= en
            for d in t.dst:
                self.X[d] |= en
                self._entered[d] |= en
                self.done[d] &= ~en
            fired.append(t.tid)
            if self.event_log is not None and self.log_transitions:
                self.event_log.emit("transition", self.clock, self.name,
                                    {"tid": t.tid, "src": t.src, "dst": t.dst,
                                     "count": int(en.sum())})
        if self.record_trace:
            self.trace.append((self.clock, tuple(self.active()), tuple(fired)))
        return fired

    # --------------------------------------------------------------- datapath
    def action_datapath(self) -> None:
        """只執行目前 active 狀態的 Action。"""
        for sid, st in self.steps.items():
            if st.action is None:
                continue
            m = self.X[sid]
            if st.mode == "P1":
                m = m & self._entered[sid]
            if m.any():
                st.action(m)
        for sid in self._entered:
            self._entered[sid][:] = False
        self.clock += 1

    def tick(self) -> List[str]:
        fired = self.controller()
        self.action_datapath()
        return fired

    # ---------------------------------------------------------------- helpers
    def mark_done(self, sid: str, mask: Optional[Mask] = None) -> None:
        """由 Action 呼叫：產生完成事件（作為轉移條件）。"""
        if mask is None:
            self.done[sid][:] = True
        else:
            self.done[sid] |= mask

    def active(self) -> List[str]:
        return [sid for sid, x in self.X.items() if x.any()]

    def is_active(self, sid: str) -> bool:
        return bool(self.X[sid].any())

    def to_mermaid(self) -> str:
        lines = ["flowchart TD"]
        for sid, st in self.steps.items():
            label = f"{sid}<br/>{st.desc}" if st.desc else sid
            lines.append(f'    {sid}{"[[" if st.initial else "["}"{label}"{"]]" if st.initial else "]"}')
        for t in self.transitions:
            tnode = f"{t.tid}"
            lines.append(f'    {tnode}{{{{"{t.tid}: {t.desc or "cond"}"}}}}')
            for s in t.src:
                lines.append(f"    {s} --> {tnode}")
            for d in t.dst:
                lines.append(f"    {tnode} --> {d}")
        return "\n".join(lines)

    def state_table(self) -> str:
        """輸出 Markdown 狀態表（Step / Action / 轉移條件 / 下一狀態）。"""
        rows = ["| Step | IDEF0 | Action (mode) | 轉移 | 條件 | 下一狀態 |",
                "|---|---|---|---|---|---|"]
        for sid, st in self.steps.items():
            act = st.action.__name__ if st.action else "—"
            outs = [t for t in self.transitions if sid in t.src]
            if not outs:
                rows.append(f"| {sid}{' (init)' if st.initial else ''} | {st.idef0} | {act} ({st.mode}) | — | (terminal) | — |")
            for t in outs:
                rows.append(f"| {sid}{' (init)' if st.initial else ''} | {st.idef0} | {act} ({st.mode}) | {t.tid} | {t.desc} | {', '.join(t.dst)} |")
        return "\n".join(rows)


def t_desc_ok(g: Grafcet, sid: str) -> bool:
    return all(t.desc for t in g.transitions if sid in t.src)


class SuperLoop:
    """裸機 super-loop 的模擬：每個 clock 先執行所有 Controller，再執行所有 Action。

    所有 GRAFCET 上電後同時存在（MIAT 4.4.4）：子控制器以上層狀態變數作為
    啟動條件，以自身完成狀態回饋上層，而非 call-return。
    """

    def __init__(self, grafcets: Sequence[Grafcet]):
        self.grafcets = list(grafcets)
        self.clock = 0

    def reset(self) -> None:
        for g in self.grafcets:
            g.reset()
        self.clock = 0

    def tick(self) -> None:
        for g in self.grafcets:
            g.controller()
        for g in self.grafcets:
            g.action_datapath()
        self.clock += 1

    def run(self, until: Callable[[], bool], max_ticks: int = 10_000_000) -> int:
        while not until():
            if self.clock >= max_ticks:
                raise GrafcetError(f"超過 max_ticks={max_ticks}，可能有 deadlock 或條件永不成立")
            self.tick()
        return self.clock
