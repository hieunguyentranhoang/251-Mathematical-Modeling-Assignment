from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Any, Set, List

try:
    from utils import Timer, TaskStats
except Exception:
    import time
    @dataclass
    class TaskStats:
        seconds: float
        peak_mb: float = 0.0
        extra: Optional[dict] = None
    class Timer:
        def __init__(self, enable_memory=True):
            self.enable_memory = enable_memory
            self.seconds = 0.0
            self.peak_mb = 0.0
        def __enter__(self):
            import time as _t
            self._t0 = _t.perf_counter()
            return self
        def __exit__(self, *a):
            import time as _t
            self.seconds = _t.perf_counter() - self._t0

# --- Libraries ---
HAVE_DD = True
try:
    from dd import autoref as _bdd
except Exception:
    HAVE_DD = False
    _bdd = None 

# "Combine ILP"
HAVE_PULP = True
try:
    import pulp
except Exception:
    HAVE_PULP = False


@dataclass
class DeadlockResult:
    found: bool
    marking: Optional[Set[str]]
    method: str
    stats: TaskStats


def _unpack_bdd(bdd_res=None, bdd_manager=None, reachable_bdd=None, place_vars=None, count=None):
    if bdd_res is not None:
        mgr = getattr(bdd_res, "bdd_manager", None)
        R   = getattr(bdd_res, "reachable_bdd", None)
        X   = getattr(bdd_res, "place_vars", None)
        if mgr is not None and R is not None and X is not None:
            return mgr, R, X
        if isinstance(bdd_res, dict):
            mgr = bdd_res.get("bdd_manager") or bdd_res.get("manager")
            R   = bdd_res.get("reachable_bdd") or bdd_res.get("bdd")
            X   = bdd_res.get("place_vars")
            if mgr is not None and R is not None and X is not None:
                return mgr, R, X
    if bdd_manager is not None and reachable_bdd is not None and place_vars is not None:
        return bdd_manager, reachable_bdd, place_vars
    return None, None, None


def _enabled_t(bdd, X: Dict[str, str], pre_set: Set[str]):
    En = bdd.true
    for p in pre_set:
        En &= bdd.var(X[p])
    return En


def _verify_with_ilp(pn, candidate_marking: Set[str]) -> bool:
    if not HAVE_PULP:
        return True 

    try:
        prob = pulp.LpProblem("Deadlock_Check", pulp.LpMinimize)
        is_dead = True
        
        for t in pn.transitions:
            pre_t = pn.pre.get(t, set())
            if not pre_t: 
                continue 
            
            current_sum = sum(1 for p in pre_t if p in candidate_marking)
            
            threshold = len(pre_t)
            
            if current_sum >= threshold:
                is_dead = False
                break
        
        return is_dead
        
    except Exception:
        return True 


def run_deadlock(pn,
                 bdd_res=None,
                 bdd=None,
                 bdd_manager=None,
                 reachable_bdd=None,
                 place_vars=None,
                 confirm_with_ilp: bool = True,
                 track_memory: bool = True):
    
    # 1. Kiểm tra thư viện
    if not HAVE_DD:
        return {"found": False, "marking": None, "method": "None", 
                "stats": TaskStats(seconds=0.0, peak_mb=0.0)}

    mgr, R, X = _unpack_bdd(bdd_res, bdd_manager, reachable_bdd, place_vars)
    if R is None or X is None:
        return {"found": False, "marking": None, "method": "None",
                "stats": TaskStats(seconds=0.0, peak_mb=0.0)}

    with Timer(enable_memory=track_memory) as tm:
        Dead_BDD = R 
        
        for t in pn.transitions:
            pre_t = set(pn.pre.get(t, set()))
            if not pre_t:
                continue 
                
            En_t = _enabled_t(mgr, X, pre_t)
            
            Dead_BDD &= ~En_t

        model = mgr.pick(Dead_BDD)
        
        if model is None:
            return {"found": False, "marking": None, "method": "BDD+ILP",
                    "stats": TaskStats(seconds=tm.seconds, peak_mb=tm.peak_mb)}

        Mset = sorted([p for p, vname in X.items() if bool(model.get(vname, 0))])
        candidate_set = set(Mset)

        is_valid_deadlock = True
        method_str = "BDD"
        
        if confirm_with_ilp and HAVE_PULP:
            method_str = "BDD + ILP Verification"
            if not _verify_with_ilp(pn, candidate_set):
                is_valid_deadlock = False
        
        if not is_valid_deadlock:
             return {"found": False, "marking": None, "method": method_str,
                    "stats": TaskStats(seconds=tm.seconds, peak_mb=tm.peak_mb)}

    return {"found": True, "marking": Mset, "method": method_str,
            "stats": TaskStats(seconds=tm.seconds, peak_mb=tm.peak_mb)}


def deadlock_detection(*a, **k): return run_deadlock(*a, **k)
def find_deadlock(*a, **k):      return run_deadlock(*a, **k)
def run(*a, **k):                return run_deadlock(*a, **k)