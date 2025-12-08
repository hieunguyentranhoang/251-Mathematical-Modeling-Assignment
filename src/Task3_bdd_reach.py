from __future__ import annotations
from typing import Dict, Set, Iterable, Optional, Any, List, Tuple
from dataclasses import dataclass

# === Timer + TaskStats abstraction ===
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


# --- dd.autoref (BDD backend) ----
HAVE_DD = True
try:
    from dd import autoref as _bdd  
except Exception:  
    HAVE_DD = False
    _bdd = None  

def _equiv(u, v):
    """BDD equivalence: u <-> v  ==  (u & v) | (~u & ~v)."""
    return (u & v) | (~u & ~v)


def _declare_vars(bdd: "_bdd.BDD", X: Dict[str, str], Xp: Dict[str, str]) -> None:
    """
    Declare variables in *interleaved* order x_p, xp_p, x_q, xp_q, ...
    This keeps current/next variables close in the ordering.
    """
    sorted_places = sorted(X.keys())
    names: List[str] = []
    for p in sorted_places:
        names.append(X[p])     # current
        names.append(Xp[p])    # next
    bdd.declare(*names)


def _cube_from_marking(bdd: "_bdd.BDD", X: Dict[str, str], marking: Set[str]):
    """Return a BDD cube for a given 1-safe marking on CURRENT vars."""
    u = bdd.true
    for p, vname in X.items():
        v = bdd.var(vname)
        u &= (v if p in marking else ~v)
    return u


#Transition relations and image computation

def _t_relation_for_transition(
    bdd: "_bdd.BDD",
    X: Dict[str, str],
    Xp: Dict[str, str],
    pre_set: Set[str],
    post_set: Set[str],
):
    """
    Local transition relation R_t(X,X') for a 1-safe net
    """
    # Enabling condition: all input places marked
    En = bdd.true
    for p in pre_set:
        En &= bdd.var(X[p])

    # Update (next-state constraints)
    Upd = bdd.true
    for p in X.keys():
        v = bdd.var(X[p])      # x_p
        vp = bdd.var(Xp[p])    # x'_p
        in_pre = p in pre_set
        in_post = p in post_set

        if (not in_pre) and (not in_post):
            # Unaffected place
            Upd &= _equiv(vp, v)
        elif (not in_pre) and in_post:
            # Only produced
            Upd &= vp
        elif in_pre and (not in_post):
            # Only consumed
            Upd &= ~vp
        else:
            # Self-loop
            Upd &= _equiv(vp, v)

    return En & Upd


def _build_relation(
    bdd: "_bdd.BDD",
    pn: "PetriNet",
    X: Dict[str, str],
    Xp: Dict[str, str],
):
    """
    Build the *monolithic* transition relation T(X,X') = OR_t R_t(X,X').
    """
    T = bdd.false
    for t in pn.transitions:
        pre = set(pn.pre[t])
        post = set(pn.post[t])
        T |= _t_relation_for_transition(bdd, X, Xp, pre, post)
    return T


def _build_local_relations(
    bdd: "_bdd.BDD",
    pn: "PetriNet",
    X: Dict[str, str],
    Xp: Dict[str, str],
):
    """
    Build a list of local relations R_t(X,X'), one per transition.
    """
    rels = []
    for t in pn.transitions:
        pre = set(pn.pre[t])
        post = set(pn.post[t])
        rels.append(_t_relation_for_transition(bdd, X, Xp, pre, post))
    return rels


def _image_via_relation(
    bdd: "_bdd.BDD",
    R,
    T,
    X: Dict[str, str],
    Xp: Dict[str, str],
):
    """
    Image computation based on the monolithic transition relation T(X,X')
    """
    if T is None:
        return bdd.false

    qvars = set(X.values())  
    img_xp = bdd.exist(qvars, R & T)

    ren = {Xp[p]: bdd.var(X[p]) for p in X.keys()}
    img = bdd.let(ren, img_xp)
    return img


def _image_topological_rel(
    bdd: "_bdd.BDD",
    R,
    rels: List[Any],
    X: Dict[str, str],
    Xp: Dict[str, str],
):
    """
    Topological image computation using per-transition relations R_t(X,X')
    """
    if not rels:
        return bdd.false

    qvars = set(X.values())
    ren = {Xp[p]: bdd.var(X[p]) for p in X.keys()}

    img = bdd.false
    for Rt in rels:
        part = R & Rt
        if part is bdd.false:
            continue
        img_xp_t = bdd.exist(qvars, part)
        if img_xp_t is bdd.false:
            continue
        img_t = bdd.let(ren, img_xp_t)
        img |= img_t
    return img


# Counting satisfying assignments over CURRENT variables

def _count_current(bdd, u, X: Dict[str, str]) -> int:
    """
    Count the number of satisfying assignments of BDD u over the current
    variables X.
    """
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

    # 2) fallback: pick-&-block on current-variable cube 
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


# === Result container ==

@dataclass
class BDDReachResult:
    reachable_bdd: Any          
    count: Optional[int]
    stats: TaskStats
    place_vars: Dict[str, str]  # map place -> current var name
    next_vars: Dict[str, str]   # map place -> next var name
    bdd_manager: Any
    iterations: int = 0
    image_mode: str = "topological" 


# === Main algorithm =====

def symbolic_fixpoint(
    pn: "PetriNet",
    *,
    track_memory: bool = True,
    max_iters: Optional[int] = None,
    image_mode: str = "topological",
) -> BDDReachResult:
    if not HAVE_DD:
        raise RuntimeError("dd.autoref not available; please install `dd` package")

    bdd = _bdd.BDD()

    # 1) allocate variables for all places
    X: Dict[str, str] = {p: f"x_{p}" for p in pn.places}
    Xp: Dict[str, str] = {p: f"xp_{p}" for p in pn.places}
    _declare_vars(bdd, X, Xp)

    # 2) initial set R
    R = _cube_from_marking(bdd, X, set(pn.initial_marking))

    # 3) prepare relations for the chosen image mode(s)
    T = None
    if image_mode in ("relation", "hybrid"):
        T = _build_relation(bdd, pn, X, Xp)

    local_rels: List[Any] = []
    if image_mode in ("topological", "hybrid"):
        local_rels = _build_local_relations(bdd, pn, X, Xp)

    # 4) fixpoint iteration
    iters = 0
    with Timer(enable_memory=track_memory) as tm:
        while True:
            if max_iters is not None and iters >= max_iters:
                break

            img = bdd.false

            if image_mode in ("relation", "hybrid"):
                img_rel = _image_via_relation(bdd, R, T, X, Xp)
                img |= img_rel

            if image_mode in ("topological", "hybrid"):
                img_top = _image_topological_rel(bdd, R, local_rels, X, Xp)
                img |= img_top

            if img is bdd.false:
                print(f"  [BDD] Iteration {iters + 1}: Fixed point reached (No image).")
                break

            new_states = img & ~R
            if new_states == bdd.false:
                print(f"  [BDD] Iteration {iters + 1}: Fixed point reached (No new states).")
                break

            try:
                count_new = _count_current(bdd, new_states, X)
            except Exception:
                count_new = "?"
            print(f"  [BDD] Iteration {iters + 1}: Found {count_new} new states")

            R |= new_states
            iters += 1

    # 5) count reachable markings (current vars only)
    try:
        c = _count_current(bdd, R, X)
    except Exception:
        c = None

    stats = TaskStats(
        seconds=getattr(tm, "seconds", 0.0),
        peak_mb=getattr(tm, "peak_mb", 0.0),
        extra={"iterations": iters, "image_mode": image_mode},
    )

    return BDDReachResult(
        reachable_bdd=R,
        count=c,
        stats=stats,
        place_vars=X,
        next_vars=Xp,
        bdd_manager=bdd,
        iterations=iters,
        image_mode=image_mode,
    )

# === CLI entrypoints ========
def bdd_reachability(pn, **kwargs):
    res = symbolic_fixpoint(pn, **kwargs)
    return {
        "bdd": res.reachable_bdd,
        "count": res.count,
        "iterations": res.iterations,
        "result": res,  # full object for downstream tasks (deadlock / optimize)
    }


def run_bdd(pn, **kwargs):
    return bdd_reachability(pn, **kwargs)


def run(pn, **kwargs):
    return bdd_reachability(pn, **kwargs)


if __name__ == "__main__":  
    try:
        from model import PetriNet  
        print("Task3_bdd_reach: module loaded successfully.")
    except Exception:
        print("Task3_bdd_reach: loaded (no model imported).")
