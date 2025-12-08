"""
Task6_cli.py — Orchestration CLI for CO2011 Petri Net Analyzer
- Task 1: PNML parse
- Task 2: Explicit reachability
- Task 3: BDD reachability (Targeting Task3_bdd_reach.py with topological support)
- Task 4: Deadlock (BDD + optional ILP confirm)
- Task 5: Optimization over reachable markings
"""

from __future__ import annotations
import argparse
import inspect
import sys
import time
import tracemalloc
from typing import Any, Iterable, List, Optional

def ts() -> str:
    return time.strftime('%Y-%m-%d %H:%M:%S')


def info(msg: str) -> None:
    print(f"{ts()} | INFO | {msg}")


def start_mem(enable: bool) -> None:
    if enable:
        tracemalloc.start()


def peak_mb(enable: bool) -> float:
    if not enable:
        return 0.0
    _, peak = tracemalloc.get_traced_memory()
    return peak / (1024.0 * 1024.0)


def human_marking(m: Optional[Iterable[str]]) -> str:
    if not m:
        return "None"
    return str(sorted(m))


def parse_weights(s: str, n_places: int) -> List[int]:
    try:
        w = [int(x.strip()) for x in s.split(',') if x.strip() != ""]
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid --weights: {e}")
    if len(w) != n_places:
        raise argparse.ArgumentTypeError(
            f"--weights length {len(w)} must equal |P|={n_places}"
        )
    return w


def find_callable(mod: Any,
                  prefer: List[str],
                  contains_any: List[str]) -> Any:
    """Return a callable from module by (1) exact names in `prefer`,
    else (2) first function whose name contains any of `contains_any`."""
    for nm in prefer:
        fn = getattr(mod, nm, None)
        if callable(fn):
            return fn
    # heuristic search
    for name, obj in inspect.getmembers(mod, inspect.isfunction):
        lower = name.lower()
        if any(key in lower for key in contains_any):
            return obj
    raise AttributeError(
        f"Entrypoint not found. Tried: {prefer} or contains {contains_any}"
    )


def call_compatible(fn: Any, **kwargs) -> Any:
    """Call fn with subset of kwargs that match its real signature."""
    sig = inspect.signature(fn)
    allowed = set(sig.parameters.keys())
    
    has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    
    if has_var_keyword:
        return fn(**kwargs)
    else:
        passing = {k: v for k, v in kwargs.items() if k in allowed}
        return fn(**passing)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--pnml", required=True, help="Path to PNML file")
    p.add_argument(
        "--objective",
        default="uniform",
        choices=["uniform", "custom"],
        help="Objective for Task 5 (uniform = all-ones weights)",
    )
    p.add_argument(
        "--confirm_ilp",
        action="store_true",
        help="Confirm deadlock by ILP feasibility (Task 4)",
    )
    p.add_argument("--logfile", default=None)
    p.add_argument(
        "--no_memory",
        action="store_true",
        help="Disable tracemalloc peak-memory reporting",
    )
    
    p.add_argument(
        "--image_mode",
        default="topological",
        choices=["topological", "relation", "hybrid"],
        help="BDD Image computation mode (default: topological)"
    )

    p.add_argument("--enumeration_threshold", type=int, default=5000)
    p.add_argument("--sample_limit", type=int, default=20000)
    
    # optimization-related flags
    p.add_argument(
        "--weights",
        type=str,
        default=None,
        help=(
            "Comma-separated integer weights; order = PetriNet.places "
            "(required if --objective custom)"
        ),
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for sampling in Task 5 (if used)",
    )
    return p


def main() -> None:
    args = build_argparser().parse_args()

    if args.logfile:
        sys.stdout = open(args.logfile, "w", buffering=1)

    try:
        from dd import autoref as _  
        print("✓ dd.autoref import successful")
    except Exception:
        print("! dd.autoref not available -- BDD Task may fail")

    mem_on = not args.no_memory
    start_mem(mem_on)

    # --- Task 1: Parse PNML
    info("Parsing PNML ...")
    from Task1_pnml_parser import parse_pnml

    t1s = time.perf_counter()
    pn = parse_pnml(args.pnml)
    t1e = time.perf_counter()
    info(f"Parsed PNML: |P|={len(pn.places)} |T|={len(pn.transitions)}")

    # Objective / weights
    weights: Optional[List[int]] = None
    if args.objective == "custom":
        if not args.weights:
            raise SystemExit("--objective custom requires --weights")
        weights = parse_weights(args.weights, len(pn.places))
        info("Using custom objective; place order & weights:")
        for p_name, w in zip(pn.places, weights):
            print(f"{p_name:>16s} : {w}")

    # Common kwargs used for cross-task compatibility
    common_kwargs = dict(
        pn=pn,
        objective=args.objective,
        weights=weights,
        enumeration_threshold=args.enumeration_threshold,
        enum_threshold=args.enumeration_threshold,  # alias
        sample_limit=args.sample_limit,
        seed=args.seed,
    )

    # --- Task 2: Explicit reachability
    info("Task 2: BFS explicit reachability ...")
    t2s = time.perf_counter()
    exp_reach = None          
    exp_count: Optional[int] = None
    exp_res_full = None       

    try:
        import Task2_explicit as T2

        fn = find_callable(
            T2,
            prefer=["run_explicit", "explicit_reachability", "bfs_reach", "run"],
            contains_any=["reach", "bfs", "explicit"],
        )
        res = call_compatible(fn, **common_kwargs)
        exp_res_full = res  

        # Normalise for logging
        if isinstance(res, dict):
            if "result" in res and res["result"] is not None:
                exp_res_full = res["result"]
            exp_reach = res.get("reach", None)
            exp_count = res.get("count", None)
            if exp_reach is None and exp_res_full is not None and hasattr(exp_res_full, "markings"):
                exp_reach = getattr(exp_res_full, "markings")
            if exp_count is None and exp_reach is not None:
                exp_count = len(exp_reach)
        elif hasattr(res, "markings"):
            exp_reach = getattr(res, "markings")
            exp_count = len(exp_reach) if exp_reach is not None else None
        else:
            exp_reach = res
            exp_count = len(exp_reach) if exp_reach is not None else None

    except Exception as e:
        info(f"[WARN] Task 2 failed: {e}")

    t2e = time.perf_counter()
    peak2 = peak_mb(mem_on)
    info(
        f"Explicit: |Reach|={exp_count}, time={t2e - t2s:.3f}s, "
        f"peak={peak2:.1f}MB"
    )

    # --- Task 3: BDD reachability
    info(f"Task 3: BDD symbolic reachability (Mode: {args.image_mode}) ...")
    t3s = time.perf_counter()
    bdd_fix = None
    bdd_count = None
    bdd_iters = None
    bdd_res_full = None  

    try:
        import Task3_bdd_reach as T3

        fn = find_callable(
            T3,
            prefer=["run_bdd", "bdd_reachability", "symbolic_reachability", "run"],
            contains_any=["bdd", "reach", "fixpoint"],
        )
        
        res = call_compatible(
            fn, 
            pn=pn, 
            track_memory=not args.no_memory,
            image_mode=args.image_mode 
        )

        if isinstance(res, dict):
            bdd_fix = res.get("bdd", res.get("reachable_bdd"))
            bdd_count = res.get("count")
            bdd_iters = res.get("iterations")
            bdd_res_full = res.get("result", res)
        else:
            bdd_res_full = res
            bdd_fix = getattr(res, "reachable_bdd", None)
            bdd_count = getattr(res, "count", None)
            bdd_iters = getattr(res, "iterations", None)

    except Exception as e:
        info(f"[WARN] Task 3 failed: {e}")
        import traceback
        traceback.print_exc()

    t3e = time.perf_counter()
    peak3 = peak_mb(mem_on)
    info(
        f"BDD: |Reach|={bdd_count}, "
        f"iters={bdd_iters if bdd_iters is not None else 'n/a'}, "
        f"time={t3e - t3s:.3f}s, peak={peak3:.1f}MB"
    )

    # --- Task 4: Deadlock detection
    info("Task 4: Deadlock detection (BDD + ILP) ...")
    t4s = time.perf_counter()
    deadlock_found = False
    deadlock_mark = None

    try:
        import Task4_deadlock as T4

        fn4 = find_callable(
            T4,
            prefer=["run_deadlock", "deadlock_detection", "find_deadlock", "run"],
            contains_any=["dead"],
        )
        res = call_compatible(
            fn4,
            pn=pn,
            bdd_res=bdd_res_full,
            confirm_with_ilp=args.confirm_ilp,
        )

        if isinstance(res, dict):
            deadlock_found = bool(res.get("found", False))
            deadlock_mark = res.get("marking", None)
        elif isinstance(res, tuple):
            deadlock_found = bool(res[0])
            if len(res) > 1:
                deadlock_mark = res[1]
        else:
            deadlock_found = bool(res)

    except Exception as e:
        info(f"[WARN] Task 4 failed: {e}")

    t4e = time.perf_counter()
    peak4 = peak_mb(mem_on)
    if deadlock_found:
        info(
            "Deadlock FOUND (method=BDD"
            f"{' + ILP' if args.confirm_ilp else ''}), "
            f"marking={human_marking(deadlock_mark)}, "
            f"time={t4e - t4s:.3f}s, peak={peak4:.1f}MB"
        )
    else:
        info(
            "Deadlock NOT found (method=BDD"
            f"{' + ILP' if args.confirm_ilp else ''}), "
            f"time={t4e - t4s:.3f}s, peak={peak4:.1f}MB"
        )

    # --- Task 5: Optimization
    info("Task 5: Optimization over Reach ...")
    t5s = time.perf_counter()
    opt_found = False
    opt_value = None
    opt_mark = None
    opt_method = None

    try:
        import Task5_optimize as T5

        fn5 = find_callable(
            T5,
            prefer=["optimize", "run_optimize", "optimize_over_reach", "run"],
            contains_any=["opt"],
        )
        explicit_for_opt = exp_reach if exp_reach is not None else exp_res_full

        res = call_compatible(
            fn5,
            pn=pn,
            bdd_res=bdd_res_full,
            explicit_res=explicit_for_opt,
            weights=weights,
            enumeration_threshold=args.enumeration_threshold,
            sample_limit=args.sample_limit,
            seed=args.seed,
        )

        if isinstance(res, dict):
            opt_found = bool(res.get("found", False))
            opt_value = res.get("value", None)
            opt_mark = res.get("marking", None)
            opt_method = res.get("method", None)
        elif isinstance(res, tuple):
            # (found, value, marking, method)
            opt_found = bool(res[0])
            if len(res) > 1:
                opt_value = res[1]
            if len(res) > 2:
                opt_mark = res[2]
            if len(res) > 3:
                opt_method = res[3]
        else:
            opt_found = bool(res)

    except Exception as e:
        info(f"[WARN] Task 5 failed: {e}")

    t5e = time.perf_counter()
    peak5 = peak_mb(mem_on)

    if opt_found:
        info(
            f"Optimization: best={opt_value} at {human_marking(opt_mark)}, "
            f"(method={opt_method if opt_method else 'n/a'}), "
            f"time={t5e - t5s:.3f}s, peak={peak5:.1f}MB"
        )
    else:
        info(
            "Optimization: found=False "
            f"(method={opt_method if opt_method else 'n/a'}), "
            f"time={t5e - t5s:.3f}s, peak={peak5:.1f}MB"
        )

    # --- SUMMARY TABLE ---
    print("\n" + "="*85)
    print(f" RESULTS SUMMARY | Model: {args.pnml}")
    print("="*85)
    
 
    header = f"{'Task':<5} | {'Component':<20} | {'Result':<30} | {'Time (s)':<10} | {'Mem (MB)':<10}"
    print(header)
    print("-" * 85)

    # Task 1: Parse
    res_t1 = f"|P|={len(pn.places)}, |T|={len(pn.transitions)}"
    print(f"{'1':<5} | {'PNML Parsing':<20} | {res_t1:<30} | {t1e - t1s:<10.4f} | {'-':<10}")

    # Task 2: Explicit
    res_t2 = f"Count={exp_count}" if exp_count is not None else "Failed"
    print(f"{'2':<5} | {'Explicit Reach':<20} | {res_t2:<30} | {t2e - t2s:<10.4f} | {peak2:<10.2f}")

    # Task 3: BDD
    res_t3 = f"Count={bdd_count}" if bdd_count is not None else "Failed"
    if bdd_iters is not None: res_t3 += f" (iters={bdd_iters})"
    
    t3_name = f"BDD ({args.image_mode[:4]})" 
    print(f"{'3':<5} | {t3_name:<20} | {res_t3:<30} | {t3e - t3s:<10.4f} | {peak3:<10.2f}")

    # Task 4: Deadlock
    status_t4 = "FOUND" if deadlock_found else "None"
    res_t4 = f"{status_t4}"
    if deadlock_found and deadlock_mark:
        m_str = human_marking(deadlock_mark)
        if len(m_str) > 20: m_str = m_str[:17] + "..."
        res_t4 += f" @ {m_str}"
    print(f"{'4':<5} | {'Deadlock Check':<20} | {res_t4:<30} | {t4e - t4s:<10.4f} | {peak4:<10.2f}")

    # Task 5: Optimize
    if opt_found:
        res_t5 = f"Max={opt_value}"
    else:
        res_t5 = "Not found/No solution"
    print(f"{'5':<5} | {'Optimization':<20} | {res_t5:<30} | {t5e - t5s:<10.4f} | {peak5:<10.2f}")

    print("="*85)


if __name__ == "__main__":
    main()
