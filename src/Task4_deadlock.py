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
    """Tạo công thức BDD kiểm tra xem một transition có được kích hoạt không."""
    En = bdd.true
    for p in pre_set:
        En &= bdd.var(X[p])
    return En


#HYBRID: ILP STATE EQUATION
def _verify_reachability_with_ilp(pn, candidate_marking: Set[str]) -> bool:
    """
    Sử dụng ILP để giải phương trình trạng thái: M = M0 + C * sigma
    Để chứng minh tồn tại vector bắn (sigma) dẫn từ M0 đến Candidate M.
    """
    if not HAVE_PULP:
        return True 

    try:
        prob = pulp.LpProblem("State_Equation_Check", pulp.LpMinimize)
        
        sigma_vars = {}
        for t in pn.transitions:
            sigma_vars[t] = pulp.LpVariable(f"sigma_{t}", lowBound=0, cat=pulp.LpInteger)


        prob += pulp.lpSum(sigma_vars.values())

        
        for p in pn.places:
            m0_val = 1 if p in pn.initial_marking else 0
            
            m_target_val = 1 if p in candidate_marking else 0
            
            token_flow = []
            for t in pn.transitions:
                weight_post = 1 if p in pn.post.get(t, []) else 0
                weight_pre  = 1 if p in pn.pre.get(t, []) else 0
                
                incidence_val = weight_post - weight_pre
                
                if incidence_val != 0:
                    token_flow.append(incidence_val * sigma_vars[t])
            
            prob += (m0_val + pulp.lpSum(token_flow) == m_target_val, f"Eq_Place_{p}")

        status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
        
        if status == pulp.LpStatusOptimal:
            return True
        else:
            return False

    except Exception as e:
        print(f"Warning: ILP check failed due to error: {e}")
        return True 


def run_deadlock(pn,
                 bdd_res=None,
                 bdd=None,
                 bdd_manager=None,
                 reachable_bdd=None,
                 place_vars=None,
                 confirm_with_ilp: bool = True,
                 track_memory: bool = True):
    
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
            return {"found": False, "marking": None, "method": "BDD (Hybrid)",
                    "stats": TaskStats(seconds=tm.seconds, peak_mb=tm.peak_mb)}

        Mset = sorted([p for p, vname in X.items() if bool(model.get(vname, 0))])
        candidate_set = set(Mset)

        is_valid_deadlock = True
        method_str = "BDD Only"
        
        # 3. ILP PHASE: Verification (Hybrid Check)
        if confirm_with_ilp and HAVE_PULP:
            method_str = "Hybrid (BDD + ILP State Eq)"
            is_reachable_structurally = _verify_reachability_with_ilp(pn, candidate_set)
            
            if not is_reachable_structurally:
                is_valid_deadlock = False
        
        if not is_valid_deadlock:
             return {"found": False, "marking": None, "method": method_str,
                    "stats": TaskStats(seconds=tm.seconds, peak_mb=tm.peak_mb)}

    return {"found": True, "marking": Mset, "method": method_str,
            "stats": TaskStats(seconds=tm.seconds, peak_mb=tm.peak_mb)}


def deadlock_detection(*a, **k): return run_deadlock(*a, **k)
def find_deadlock(*a, **k):      return run_deadlock(*a, **k)
def run(*a, **k):                return run_deadlock(*a, **k)
