"""
事件契約（MIAT 3.4.3）與事件紀錄
================================

每個事件都帶有 event_type / timestamp / source_module / payload / priority /
policy。用途：

* 主題二（責任追溯）：每一次費洛蒙沉積、物資拿起／放下都留下「誰、何時、
  做了什麼、依據什麼機率」的紀錄，事後可做歸因分析。
* 主題三（可逆性）：外部擾動（封路、移除螞蟻、切斷介質）以高優先權事件
  進入系統，在 GRAFCET 的週期邊界被處理。
"""
from __future__ import annotations

import csv
import heapq
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Optional


# 優先權：安全／環境事件 > 任務事件 > 紀錄事件
PRIORITY_SAFETY = 100
PRIORITY_ENV = 50
PRIORITY_MISSION = 10
PRIORITY_LOG = 0


@dataclass(frozen=True)
class Event:
    event_type: str
    timestamp: int                 # 模擬 clock 或 cycle
    source_module: str             # IDEF0 模組或 agent id，例如 "A4" / "ant:12"
    payload: Dict[str, Any] = field(default_factory=dict)
    priority: int = PRIORITY_LOG
    policy: str = "merge"          # interrupt / merge / ignore

    def to_row(self) -> Dict[str, Any]:
        d = asdict(self)
        d["payload"] = json.dumps(self.payload, ensure_ascii=False, default=_np_default)
        return d


def _np_default(o):
    try:
        return o.tolist()
    except AttributeError:
        return str(o)


class EventLog:
    """只增不改（append-only）的事件紀錄。"""

    def __init__(self, enabled: bool = True, keep_types: Optional[Iterable[str]] = None):
        self.enabled = enabled
        self.keep_types = set(keep_types) if keep_types else None
        self.events: List[Event] = []

    def emit(self, event_type: str, timestamp: int, source_module: str,
             payload: Optional[Dict[str, Any]] = None, priority: int = PRIORITY_LOG,
             policy: str = "merge") -> Optional[Event]:
        if not self.enabled:
            return None
        if self.keep_types is not None and event_type not in self.keep_types:
            return None
        ev = Event(event_type, int(timestamp), source_module, payload or {}, priority, policy)
        self.events.append(ev)
        return ev

    def filter(self, event_type: Optional[str] = None, source_prefix: Optional[str] = None) -> List[Event]:
        out = self.events
        if event_type:
            out = [e for e in out if e.event_type == event_type]
        if source_prefix:
            out = [e for e in out if e.source_module.startswith(source_prefix)]
        return out

    def __len__(self) -> int:
        return len(self.events)

    def to_csv(self, path: str) -> None:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["event_type", "timestamp", "source_module",
                                              "payload", "priority", "policy"])
            w.writeheader()
            for e in self.events:
                w.writerow(e.to_row())

    def to_jsonl(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for e in self.events:
                f.write(json.dumps(e.to_row(), ensure_ascii=False) + "\n")


class EventQueue:
    """外部事件排程佇列：依 (timestamp, -priority) 取出到期事件。"""

    def __init__(self, events: Iterable[Event] = ()):
        self._h: List = []
        self._n = 0
        for e in events:
            self.push(e)

    def push(self, e: Event) -> None:
        heapq.heappush(self._h, (e.timestamp, -e.priority, self._n, e))
        self._n += 1

    def pop_due(self, now: int) -> List[Event]:
        """取出所有 timestamp <= now 的事件；同時到期者依 priority 由高到低。

        policy='ignore' 的過期事件（timestamp < now）會被丟棄。
        """
        out = []
        while self._h and self._h[0][0] <= now:
            ts, _, _, e = heapq.heappop(self._h)
            if e.policy == "ignore" and ts < now:
                continue
            out.append(e)
        return out

    def __len__(self) -> int:
        return len(self._h)
