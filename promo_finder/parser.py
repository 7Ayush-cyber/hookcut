"""Parse Kuku-style timestamped screenplays into a flat list of events.

Deterministic, no AI. Every timestamp the tool ever outputs comes from here,
so an LLM can never invent a time that isn't in the script.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

TS_RE = re.compile(r"(\d+):(\d+(?:\.\d+)?)")
BLOCK_HEAD_RE = re.compile(r"^\*\*(?P<head>.+?):\*\*\s*$")
RANGE_RE = re.compile(r"Start:\s*\(([\d:.]+)\)\s*\|\s*End:\s*\(([\d:.]+)\)")


def to_sec(ts: str) -> float:
    m = TS_RE.search(ts)
    if not m:
        raise ValueError(f"bad timestamp {ts!r}")
    return int(m.group(1)) * 60 + float(m.group(2))


def fmt(sec: float) -> str:
    sec = max(0.0, sec)
    m = int(sec // 60)
    return f"{m:02d}:{sec - m * 60:05.2f}"


@dataclass
class Event:
    kind: str            # DIALOGUE | ACTION | CAMERA | SFX
    start: float
    end: float
    text: str
    speaker: str = ""
    tone: str = ""        # e.g. "Cold, Commanding"
    scene: int = 0

    def label(self) -> str:
        if self.kind == "DIALOGUE":
            t = f" ({self.tone})" if self.tone else ""
            return f"{self.speaker}{t}: {self.text}"
        return f"[{self.kind}] {self.text}"


@dataclass
class Episode:
    number: int
    title: str
    logline: str
    continuity: str
    duration: float
    events: list[Event] = field(default_factory=list)
    scenes: list[tuple[int, float, float, str]] = field(default_factory=list)  # (n, start, end, heading)
    source: str = ""

    @property
    def story_events(self) -> list[Event]:
        """Events that carry story (everything except camera directions)."""
        return [e for e in self.events if e.kind != "CAMERA"]


def parse_episode(text: str, source: str = "") -> Episode:
    lines = text.splitlines()
    meta = {}
    for ln in lines[:10]:
        for key in ("EPISODE NUMBER", "EPISODE LOGLINE", "CONTINUITY NOTES"):
            if ln.startswith(key + ":"):
                meta[key] = ln.split(":", 1)[1].strip()
    num_m = re.search(r"\d+", meta.get("EPISODE NUMBER", "") or source)
    ep = Episode(
        number=int(num_m.group()) if num_m else 0,
        title=meta.get("EPISODE NUMBER", source),
        logline=meta.get("EPISODE LOGLINE", ""),
        continuity=meta.get("CONTINUITY NOTES", ""),
        duration=0.0,
        source=source,
    )

    scene_no, scene_start, heading = 0, 0.0, ""
    i = 0
    while i < len(lines):
        ln = lines[i].strip()
        low = ln.lower()
        if re.match(r"^scene\s+\d+", low):
            scene_no = int(re.search(r"\d+", ln).group())
        elif low.startswith("scene heading:"):
            heading = ln.split(":", 1)[1].strip()
        elif low.startswith("scene start:"):
            scene_start = to_sec(ln.split(":", 1)[1])
            if ep.scenes and ep.scenes[-1][0] == scene_no:   # unnumbered scene block
                scene_no += 1
        elif low.startswith("scene end:"):
            s_end = to_sec(ln.split(":", 1)[1])
            ep.scenes.append((scene_no, scene_start, s_end, heading))
            ep.duration = max(ep.duration, s_end)
        else:
            m = BLOCK_HEAD_RE.match(ln)
            if m and i + 1 < len(lines):
                rng = RANGE_RE.search(lines[i + 1])
                if rng:
                    body = []
                    j = i + 2
                    while j < len(lines) and lines[j].strip() and not BLOCK_HEAD_RE.match(lines[j].strip()):
                        body.append(lines[j].strip())
                        j += 1
                    head = m.group("head").strip()
                    content = " ".join(body).strip()
                    start, end = to_sec(rng.group(1)), to_sec(rng.group(2))
                    if head == "CAMERA":
                        kind, spk, tone = "CAMERA", "", ""
                    elif head == "ACTION":
                        kind, spk, tone = "ACTION", "", ""
                    elif head.startswith("SOUND"):
                        kind, spk, tone = "SFX", "", ""
                    else:
                        kind = "DIALOGUE"
                        tm = re.search(r"\(([^()]*)\)\s*$", head)
                        tone = tm.group(1) if tm else ""
                        spk = re.sub(r"\(.*?\)", "", head).strip()
                        content = content.strip('"“”')
                    ep.events.append(Event(kind, start, end, content, spk, tone, scene_no))
                    ep.duration = max(ep.duration, end)
                    i = j
                    continue
        i += 1
    ep.events.sort(key=lambda e: (e.start, e.end))
    return ep


def load_show(folder: str | Path) -> list[Episode]:
    folder = Path(folder)
    eps = [parse_episode(p.read_text(encoding="utf-8", errors="replace"), p.name)
           for p in sorted(folder.glob("*.txt"))]
    return sorted(eps, key=lambda e: e.number)
