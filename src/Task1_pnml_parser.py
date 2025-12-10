from __future__ import annotations
import xml.etree.ElementTree as ET
from typing import Dict, Set, List
from model import PetriNet

def _local(tag: str) -> str:
    return tag.split("}", 1)[1] if "}" in tag else tag

def parse_pnml(file_path: str) -> PetriNet:
    """
    Read a 1-safe Petri net from PNML and construct internal representation:
      - places with initial marking
      - transitions
      - flow relations (pre, post)
    Consistency: every arc must reference existing nodes.
    Assumption: input models are 1-safe (per assignment). 
    """
    tree = ET.parse(file_path)
    root = tree.getroot()

    def iter_local(el, name):
        for x in el.iter():
            if _local(x.tag) == name:
                yield x

    places: List[str] = []
    transitions: List[str] = []
    arcs: List[tuple[str, str]] = []
    place_nodes: Dict[str, ET.Element] = {}

    init_mark: Set[str] = set()
    for p in iter_local(root, "place"):
        pid = p.attrib.get("id")
        if not pid:
            continue
        places.append(pid)
        place_nodes[pid] = p
        init_val = 0
        for im in p:
            if _local(im.tag).lower() in {"initialmarking", "initmarking"}:
                text_val = None
                for sub in im.iter():
                    if _local(sub.tag).lower() in {"text", "value", "marking"} and (sub.text or "").strip():
                        text_val = sub.text.strip()
                if text_val is None and (im.text or "").strip():
                    text_val = im.text.strip()
                if text_val is not None:
                    try:
                        init_val = int(float(text_val))
                    except Exception:
                        init_val = 0
        if init_val >= 1:
            init_mark.add(pid)

    # transitions
    trans_nodes: Dict[str, ET.Element] = {}
    for t in iter_local(root, "transition"):
        tid = t.attrib.get("id")
        if tid:
            transitions.append(tid)
            trans_nodes[tid] = t

    # arcs
    for a in iter_local(root, "arc"):
        s = a.attrib.get("source")
        t = a.attrib.get("target")
        if s and t:
            arcs.append((s, t))

    # consistency 
    nodes = set(places) | set(transitions)
    for s, t in arcs:
        if s not in nodes or t not in nodes:
            raise ValueError(f"Inconsistent PNML: arc {s}->{t} references non-existing node.")

    # pre/post
    pre = {t: set() for t in transitions}
    post = {t: set() for t in transitions}
    for s, t in arcs:
        if s in places and t in transitions:
            pre[t].add(s)
        elif s in transitions and t in places:
            post[s].add(t)
        else:
            # ignore non P/T arcs
            pass

    return PetriNet(places, transitions, pre, post, frozenset(init_mark))
