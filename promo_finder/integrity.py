"""Parse-integrity report: makes silent parsing failures visible (audit finding D-1)."""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from .parser import BLOCK_HEAD_RE, RANGE_RE, Episode

INJECTION_RE = re.compile(
    r"(ignore (all |any )?(previous|prior|above) (instructions|rules)|system prompt|you are now|"
    r"disregard (the|all)|as an ai|return (a )?score of|set (the )?(hook|cliffhanger)_score|"
    r"\bassistant\s*:|\buser\s*:|</?(system|instructions?)>)", re.I)


def check_show(folder: str | Path, episodes: list[Episode]) -> dict:
    files = sorted(Path(folder).glob("*.txt"))
    rep = {"files": len(files), "episodes_parsed": len(episodes), "problems": [], "per_episode": []}
    by_src = {e.source: e for e in episodes}
    nums = Counter(e.number for e in episodes)
    for n, k in nums.items():
        if k > 1:
            rep["problems"].append(f"episode number {n} appears in {k} files (later file overwrites earlier)")
    if episodes:
        have = set(nums)
        missing = [n for n in range(min(have), max(have) + 1) if n not in have]
        if missing:
            rep["problems"].append(f"missing episode numbers: {missing}")
    for f in files:
        lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
        heads = [i for i, ln in enumerate(lines) if BLOCK_HEAD_RE.match(ln.strip())]
        untimed = [lines[i].strip() for i in heads if i + 1 >= len(lines) or not RANGE_RE.search(lines[i + 1])]
        ep = by_src.get(f.name)
        parsed = len(ep.events) if ep else 0
        row = {"file": f.name, "episode": ep.number if ep else None, "blocks_in_file": len(heads),
               "events_parsed": parsed, "untimed_blocks": len(untimed),
               "duration": round(ep.duration, 2) if ep else 0, "scenes": len(ep.scenes) if ep else 0}
        if ep is None or parsed == 0:
            rep["problems"].append(f"{f.name}: no events parsed")
        if untimed:
            rep["problems"].append(f"{f.name}: {len(untimed)} block(s) without a valid Start/End line "
                                   f"were skipped, e.g. {untimed[0][:60]}")
        if ep:
            bad = [e for e in ep.events if e.end < e.start]
            if bad:
                rep["problems"].append(f"{f.name}: {len(bad)} event(s) end before they start")
            for e in ep.events:
                if INJECTION_RE.search(e.text):
                    rep["problems"].append(f"{f.name} @ {e.start:.2f}s: text looks like an instruction "
                                           f"to an AI (kept as data, LLM blend disabled for clips containing it)")
        rep["per_episode"].append(row)
    rep["ok"] = not rep["problems"]
    return rep
