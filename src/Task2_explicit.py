# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from typing import Set, FrozenSet
from collections import deque
from model import PetriNet
from utils import Timer, TaskStats

@dataclass
class ExplicitReachResult:
    markings: Set[FrozenSet[str]]
    stats: TaskStats

def bfs_reachability(pn: PetriNet, track_memory: bool = True) -> ExplicitReachResult:
    """
    Enumerate all reachable markings from initial marking using BFS.  :contentReference[oaicite:5]{index=5}
    """
    with Timer(enable_memory=track_memory) as tm:
        visited: Set[FrozenSet[str]] = set([pn.initial_marking])
        q = deque([pn.initial_marking])
        while q:
            m = q.popleft()
            for t in pn.enabled_transitions(m):
                m2 = pn.fire(m, t)
                if m2 not in visited:
                    visited.add(m2)
                    q.append(m2)
    return ExplicitReachResult(
        markings=visited,
        stats=TaskStats(seconds=tm.seconds, peak_mb=tm.peak_mb, extra={"num_markings": len(visited)}),
    )
# --- SAFE WRAPPER (Task 2) ---
import inspect
from typing import Any, Dict

def _call_compat(fn, **kwargs):
    sig = inspect.signature(fn)
    return fn(**{k: v for k, v in kwargs.items() if k in sig.parameters})

def _find_core_explicit():
    prefer = ["bfs_reachability", "bfs_reach", "explicit_core"]
    for name in prefer:
        f = globals().get(name)
        if callable(f):
            return f
    for name, obj in globals().items():
        if callable(obj):
            low = name.lower()
            if "reach" in low and "run" not in low:
                return obj
    raise RuntimeError("No explicit core function found")

def run_explicit(pn, **kwargs) -> Dict[str, Any]:
    core = _find_core_explicit()
    res = _call_compat(core, pn=pn, **kwargs)
    # Normalize
    if isinstance(res, dict):
        reach = res.get("reach")
        cnt   = res.get("count", len(reach) if reach is not None else None)
    elif hasattr(res, "markings"):  # dataclass ExplicitReachResult
        reach = getattr(res, "markings")
        cnt   = len(reach) if reach is not None else None
    else:
        reach = res
        cnt   = len(reach) if reach is not None else None
    return {"reach": reach, "count": cnt}
