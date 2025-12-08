from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Any, Iterable, Set, Tuple, List
import random 

try:
    from utils import Timer, TaskStats
except Exception:
    @dataclass
    class TaskStats:
        seconds: float
        peak_mb: float = 0.0
        extra: Optional[dict] = None
    import time
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

HAVE_DD = True
try:
    from dd import autoref as _bdd
except Exception:
    HAVE_DD = False
    _bdd = None  


def _as_marking_set(explicit_res) -> Optional[Set[frozenset]]:
    if explicit_res is None:
        return None
    if hasattr(explicit_res, "markings"):
        return set(explicit_res.markings)
    if isinstance(explicit_res, (set, list, tuple)):
        return set(explicit_res)
    return None


def _unpack_bdd(bdd_res=None, bdd_manager=None, reachable_bdd=None, place_vars=None, count=None):
    if bdd_res is not None:
        bdd = getattr(bdd_res, "bdd_manager", None)
        S   = getattr(bdd_res, "reachable_bdd", None)
        X   = getattr(bdd_res, "place_vars", None)
        n   = getattr(bdd_res, "count", None)
        if bdd is not None and S is not None and X is not None:
            return bdd, S, X, n
        if isinstance(bdd_res, dict):
            bdd = bdd_res.get("bdd_manager") or bdd_res.get("manager")
            S   = bdd_res.get("reachable_bdd") or bdd_res.get("bdd")
            X   = bdd_res.get("place_vars")
            n   = bdd_res.get("count")
            if bdd is not None and S is not None and X is not None:
                return bdd, S, X, n
    if bdd_manager is not None and reachable_bdd is not None and place_vars is not None:
        return bdd_manager, reachable_bdd, place_vars, count
    return None, None, None, None


def _infer_places(pn, place_vars, explicit_V) -> Optional[List[str]]:
    if place_vars:
        return list(place_vars.keys())
    if explicit_V:
        uni = set()
        for m in explicit_V:
            uni |= set(m)
        return sorted(uni)
    if pn is not None and hasattr(pn, "places"):
        return list(pn.places)
    return None


def _coerce_weights(places_order: List[str], weights) -> Dict[str, int]:
    if weights is None:
        return {p: 1 for p in places_order}
    if isinstance(weights, dict):
        return {p: int(weights.get(p, 1)) for p in places_order}
    if isinstance(weights, (list, tuple)):
        if len(weights) != len(places_order):
            raise ValueError("weights length must equal number of places")
        return {p: int(v) for p, v in zip(places_order, weights)}
    raise TypeError("Unsupported weights type for 'weights'")


def _obj(marking: Iterable[str], W: Dict[str, int]) -> int:
    return sum(W.get(p, 0) for p in marking)


def _guided_pick(bdd, S, place_vars: Dict[str, str], weights: Dict[str, int], 
                 randomize: bool = False, epsilon: float = 0.5) -> frozenset:
    current_u = S
    chosen_places = []
    
    for p, vname in place_vars.items():
        var_node = bdd.var(vname)
        
        pos_feasible = (current_u & var_node) != bdd.false
        
        neg_feasible = (current_u & ~var_node) != bdd.false
        
        must_pick_pos = pos_feasible and not neg_feasible
        must_pick_neg = neg_feasible and not pos_feasible
        can_pick_both = pos_feasible and neg_feasible
        
        choice = False 
        
        if must_pick_pos:
            choice = True
        elif must_pick_neg:
            choice = False
        elif can_pick_both:
            if randomize and random.random() < epsilon:
                choice = random.choice([True, False])
            else:
                w = weights.get(p, 0)
                if w > 0:
                    choice = True
                elif w < 0:
                    choice = False
                else:
                    choice = random.choice([True, False])
        else:
            break 
            
        if choice:
            chosen_places.append(p)
            current_u &= var_node
        else:
            current_u &= ~var_node
            
    return frozenset(chosen_places)
# ---- main entry ------
def optimize(pn=None,
             bdd_res=None,
             explicit_res=None,
             bdd_manager=None, reachable_bdd=None, place_vars=None, count=None,
             weights: Optional[Dict[str, int]] = None,
             sample_limit: int = 1000, 
             enumeration_threshold: int = 200_000,
             seed: Optional[int] = None,
             track_memory: bool = True):
    
    if seed is not None:
        random.seed(seed)

    V = _as_marking_set(explicit_res)
    bdd, S0, X, N = _unpack_bdd(bdd_res, bdd_manager, reachable_bdd, place_vars, count)

    places_order = _infer_places(pn, X, V)
    if not places_order:
        return {
            "found": False,
            "value": None,
            "marking": None,
            "method": "n/a",
            "stats": TaskStats(seconds=0.0, peak_mb=0.0),
        }

    W = _coerce_weights(places_order, weights)

    with Timer(enable_memory=track_memory) as tm:
        best_m = None
        best_v = None
        method = "n/a"

        # CASE 1: Duyệt cạn 
        if V is not None and len(V) <= enumeration_threshold:
            method = "enumerate_explicit"
            for m in V:
                v = _obj(m, W)
                if (best_v is None) or (v > best_v):
                    best_v, best_m = v, m
        
        # CASE 2: BDD 
        elif HAVE_DD and bdd is not None and S0 is not None and X is not None:
            S = S0
            if (N is not None) and (N <= enumeration_threshold):
                method = "enumerate_bdd"
                seen = set()
                temp_S = S
                while True:
                    model = bdd.pick(temp_S)
                    if model is None:
                        break
                    m = frozenset(p for p, vname in X.items() if bool(model.get(vname, 0)))
                    
                    if m not in seen:
                        seen.add(m)
                        v = _obj(m, W)
                        if (best_v is None) or (v > best_v):
                            best_v, best_m = v, m
                    
                    cube = bdd.true
                    for p, vname in X.items():
                        v_node = bdd.var(vname)
                        cube &= (v_node if p in m else ~v_node)
                    temp_S = temp_S & ~cube
            else:
                method = "heuristic_bdd_search" 
                
                # Bước 1: Greedy Search
                m_greedy = _guided_pick(bdd, S, X, W, randomize=False)
                v_greedy = _obj(m_greedy, W)
                best_v, best_m = v_greedy, m_greedy
                
                # Bước 2: Randomized Search 
                iters = min(sample_limit, 500)
                
                for _ in range(iters):
                    m_rand = _guided_pick(bdd, S, X, W, randomize=True, epsilon=0.5)
                    v_rand = _obj(m_rand, W)
                    
                    if v_rand > best_v:
                        best_v, best_m = v_rand, m_rand
        found = best_m is not None
        return {
            "found": found,
            "value": (int(best_v) if found else None),
            "marking": (sorted(list(best_m)) if found else None),
            "method": (method if found else "n/a"),
            "stats": TaskStats(seconds=tm.seconds, peak_mb=tm.peak_mb),
        }
def run_optimize(*a, **k):        return optimize(*a, **k)
def optimize_over_reach(*a, **k): return optimize(*a, **k)
def run(*a, **k):                 return optimize(*a, **k)
