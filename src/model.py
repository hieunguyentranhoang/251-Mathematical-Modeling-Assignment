# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Set, FrozenSet, List

@dataclass
class PetriNet:
    places: List[str]
    transitions: List[str]
    pre: Dict[str, Set[str]]
    post: Dict[str, Set[str]]
    initial_marking: FrozenSet[str]

    place_index: Dict[str, int] = field(init=False)

    def __post_init__(self):
        self.place_index = {p: i for i, p in enumerate(self.places)}

    def enabled(self, marking: FrozenSet[str], t: str) -> bool:
        return self.pre.get(t, set()) <= marking

    def fire(self, marking: FrozenSet[str], t: str) -> FrozenSet[str]:
        # 1-safe semantics
        new_m = set(marking)
        for p in self.pre.get(t, set()):
            new_m.discard(p)
        for p in self.post.get(t, set()):
            new_m.add(p)
        return frozenset(new_m)

    def enabled_transitions(self, marking: FrozenSet[str]):
        return [t for t in self.transitions if self.enabled(marking, t)]
