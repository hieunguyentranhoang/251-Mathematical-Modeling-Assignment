from __future__ import annotations
import time, tracemalloc
from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class TaskStats:
    seconds: float
    peak_mb: float
    extra: Dict[str, Any] = field(default_factory=dict)

class Timer:
    """Measure wall time + peak memory (MiB) via tracemalloc."""
    def __init__(self, enable_memory: bool = True):
        self.enable_memory = enable_memory
        self._start = 0.0
        self._end = 0.0
        self._peak = 0

    def __enter__(self):
        if self.enable_memory:
            if not tracemalloc.is_tracing():
                tracemalloc.start()
            else:
                tracemalloc.clear_traces()
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._end = time.perf_counter()
        if self.enable_memory:
            _, self._peak = tracemalloc.get_traced_memory()
            pass 

    @property
    def seconds(self) -> float:
        return self._end - self._start

    @property
    def peak_mb(self) -> float:
        return (self._peak or 0) / (1024.0 * 1024.0)