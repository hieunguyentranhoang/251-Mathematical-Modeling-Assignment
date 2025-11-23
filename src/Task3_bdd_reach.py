from __future__ import annotations
from typing import Dict, Set, Iterable, Optional, Any
from dataclasses import dataclass

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
        def __init__(self, enable_memory: bool = True):
            self.enable_memory = enable_memory
            self._t0 = 0.0
            self.seconds = 0.0
            self.peak_mb = 0.0
        def __enter__(self):
            self._t0 = time.perf_counter()
            return self
        def __exit__(self, exc_type, exc, tb):
            self.seconds = time.perf_counter() - self._t0

# --- dd.autoref  
HAVE_DD = True
try:
    from dd import autoref as _bdd
except Exception:
    HAVE_DD = False
    _bdd = None  # type: ignore




# === Small BDD helpers =========
def _equiv(u, v):
    """BDD equivalence: u <-> v  ==  (u & v) | (~u & ~v)"""
    return (u & v) | (~u & ~v)


def _declare_vars(bdd: "_bdd.BDD", X: Dict[str, str], Xp: Dict[str, str]) -> None:
    """Declare variables in interleaved order: x1, xp1, x2, xp2..."""
    sorted_places = sorted(X.keys())
    names = []
    for p in sorted_places:
        names.append(X[p])   
        names.append(Xp[p])  
    bdd.declare(*names)

def _cube_from_marking(bdd: "_bdd.BDD", X: Dict[str, str], marking: Set[str]):
    """Return a BDD cube for a given marking on CURRENT vars."""
    u = bdd.true
    for p, vname in X.items():
        v = bdd.var(vname)
        u &= (v if p in marking else ~v)
    return u


def _t_relation_for_transition(
    bdd: "_bdd.BDD",
    X: Dict[str, str],
    Xp: Dict[str, str],
    pre_set: Set[str],
    post_set: Set[str],
):
    En = bdd.true
    for p in pre_set:
        En &= bdd.var(X[p])

    Upd = bdd.true
    for p in X.keys():
        v = bdd.var(X[p])    
        vp = bdd.var(Xp[p])  
        in_pre = p in pre_set
        in_post = p in post_set

        if (not in_pre) and (not in_post):
            Upd &= _equiv(vp, v)
        elif (not in_pre) and in_post:
            Upd &= vp
        elif in_pre and (not in_post):
            Upd &= ~vp
        else:
            # self-loop: keep
            Upd &= _equiv(vp, v)

    return En & Upd


def _build_relation(
    bdd: "_bdd.BDD",
    pn: "PetriNet",
    X: Dict[str, str],
    Xp: Dict[str, str],
):
    T = bdd.false
    for t in pn.transitions:
        pre = set(pn.pre[t])
        post = set(pn.post[t])
        T |= _t_relation_for_transition(bdd, X, Xp, pre, post)
    return T


# --- REPLACE your _count_current(...) by this version ---
def _count_current(bdd, u, X):
    # 1) dd versions that support care_vars
    try:
        count = 0
        care = set(X.values())
        for _ in bdd.sat_iter(u, care_vars=care):
            count += 1
        return count
    except TypeError:
        pass
    except AttributeError:
        pass

    # 2) fallback: pick-&-block on current-variable cube (safe for small/medium sets)
    S = u
    seen = 0
    while True:
        model = bdd.pick(S)
        if model is None:
            break
        cube = bdd.true
        for p, vname in X.items():
            v = bdd.var(vname)
            cube &= (v if bool(model.get(vname, 0)) else ~v)
        S = S & ~cube
        seen += 1
    return seen


# === Result container ====
@dataclass
class BDDReachResult:
    reachable_bdd: Any        # BDD node for R (current vars only)
    count: Optional[int]
    stats: TaskStats
    place_vars: Dict[str, str]   # map place -> current var name
    next_vars: Dict[str, str]    # map place -> next var name
    bdd_manager: Any
    iterations: int = 0


# === Main algorithm ======

def symbolic_fixpoint(
    pn: "PetriNet",
    *,
    track_memory: bool = True,
    max_iters: Optional[int] = None,
) -> BDDReachResult:
    if not HAVE_DD:
        raise RuntimeError("dd.autoref not available; please install `dd` package")

    bdd = _bdd.BDD()

    # 1) allocate variables for all places
    X  = {p: f"x_{p}"  for p in pn.places}
    Xp = {p: f"xp_{p}" for p in pn.places}
    _declare_vars(bdd, X, Xp)

    # 2) initial set R
    R = _cube_from_marking(bdd, X, set(pn.initial_marking))

    # 3) build T(x, x')
    T = _build_relation(bdd, pn, X, Xp)

    # 4) fixpoint
    iters = 0
    with Timer(enable_memory=track_memory) as tm:
        while True:
            if max_iters is not None and iters >= max_iters:
                break
            qvars = set(X.values())  
            img_xp = bdd.exist(qvars, R & T)

            ren = {Xp[p]: bdd.var(X[p]) for p in X.keys()}
            img = bdd.let(ren, img_xp)

            new_states = img & ~R
            if new_states == bdd.false:
                break
            R |= new_states
            iters += 1

    # 5) count (on current vars only)
    try:
        c = _count_current(bdd, R, X)
    except Exception:
        c = None

    stats = TaskStats(seconds=getattr(tm, "seconds", 0.0),
                      peak_mb=getattr(tm, "peak_mb", 0.0),
                      extra={"iterations": iters})

    return BDDReachResult(
        reachable_bdd=R,
        count=c,
        stats=stats,
        place_vars=X,
        next_vars=Xp,
        bdd_manager=bdd,
        iterations=iters,
    )


# === CLI entrypoints (stable) =====

def bdd_reachability(pn, **kwargs):
    """
    Stable wrapper for CLI (Task6_cli). Returns both summary and full object.
    """
    res = symbolic_fixpoint(pn, **kwargs)
    return {
        "bdd": res.reachable_bdd,
        "count": res.count,
        "iterations": res.iterations,
        "result": res,  # pass full object for downstream tasks (deadlock / optimize)
    }

def run_bdd(pn, **kwargs):
    return bdd_reachability(pn, **kwargs)

def run(pn, **kwargs):
    return bdd_reachability(pn, **kwargs)


# === Minimal self-test  ======
if __name__ == "__main__":
    try:
        from model import PetriNet
        # You can add a tiny PN here if desired.
        print("Task3_bdd_reach: module loaded successfully.")
    except Exception:
        print("Task3_bdd_reach: loaded (no model imported).")
