"""Generate and score promo candidates from parsed episodes (no AI)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .heuristics import REPLY_START, beat_reasons, beat_score, cliff_signals
from .parser import Episode, Event, fmt
from .scoring import cfg

MIN_LEN, MAX_LEN = 20.0, 60.0
HOOK_SECS = 3.0
CLIFF_SECS = 5.0


@dataclass
class Segment:
    episode: int
    start: float
    end: float

    @property
    def length(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict:
        return {"episode": self.episode, "start": fmt(self.start), "end": fmt(self.end),
                "start_sec": round(self.start, 2), "end_sec": round(self.end, 2)}


@dataclass
class Candidate:
    segments: list[Segment]
    hook_score: float = 0.0
    cliff_score: float = 0.0
    standalone_score: float = 0.0
    density_score: float = 0.0
    total: float = 0.0
    hook: str = ""
    cliffhanger: str = ""
    why: str = ""
    reasons: list[str] = field(default_factory=list)
    llm: dict | None = None
    id: str = ""
    rank_score: float = 0.0
    flags: list[str] = field(default_factory=list)      # diagnostics for the editor (no score effect)
    gate_fail: list[str] = field(default_factory=list)  # v2 hard minimums this promo misses

    @property
    def eligible(self) -> bool:
        return not self.gate_fail

    @property
    def length(self) -> float:
        return sum(s.length for s in self.segments)

    @property
    def episode(self) -> int:
        return self.segments[-1].episode

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": "continuous" if len(self.segments) == 1 else "multi-segment",
            "segments": [s.to_dict() for s in self.segments],
            "duration_sec": round(self.length, 1),
            "hook": self.hook, "cliffhanger": self.cliffhanger, "why_it_works": self.why,
            "scores": {"total": round(self.total, 2), "hook": round(self.hook_score, 2),
                       "cliffhanger": round(self.cliff_score, 2),
                       "standalone": round(self.standalone_score, 2),
                       "intensity": round(self.density_score, 2)},
            "rule_reasons": self.reasons,
            "eligible": self.eligible, "gate_fail": self.gate_fail, "flags": self.flags,
            "llm": self.llm,
        }


def _clip(x: float, lo=0.0, hi=10.0) -> float:
    return max(lo, min(hi, x))


def weighted(c: "Candidate") -> float:
    w = cfg().weights
    return (w["hook"] * c.hook_score + w["cliff"] * c.cliff_score
            + w["standalone"] * c.standalone_score + w["intensity"] * c.density_score)


def apply_gates(c: "Candidate") -> None:
    """v2: non-compensatory minimums. A great cliffhanger cannot buy back an opening that
    doesn't hook, and drama cannot buy back a clip a stranger can't follow."""
    g = cfg().gates
    c.gate_fail = []
    if not g:
        return
    if any(r.startswith("hook: no audible") for r in c.reasons):
        c.gate_fail.append("no audible story beat in first 3s")
    if c.hook_score < g["hook"]:
        c.gate_fail.append(f"hook {c.hook_score:.1f} < {g['hook']}")
    if c.cliff_score < g["cliff"]:
        c.gate_fail.append(f"cliffhanger {c.cliff_score:.1f} < {g['cliff']}")
    if any("feels resolved" in r for r in c.reasons):
        c.gate_fail.append("ending reads as resolved")
    if c.standalone_score < g["standalone"]:
        c.gate_fail.append(f"standalone {c.standalone_score:.1f} < {g['standalone']}")


def _hook_v2(ep: Episode, a: float, reasons: list[str]) -> tuple[float, str]:
    """Only story beats a viewer actually hears/sees in 0-3s count: the beat must start by 2.0s,
    or have >=1s of itself inside the first 3s. Camera notes never score (diagnostic only)."""
    head = [e for e in ep.story_events if a - 0.05 <= e.start < a + HOOK_SECS]
    aud = [e for e in head if e.start < a + 2.0 or (min(e.end, a + HOOK_SECS) - max(e.start, a)) >= 1.0]
    if not aud:
        reasons.append("hook: no audible line/action in first 3s")
        return 0.0, ""
    best = max(aud, key=beat_score)
    hs = beat_score(best) + 0.3 * sum(max(0.0, beat_score(e)) for e in aud if e is not best)
    reasons += ["hook " + r for r in beat_reasons(best)]
    return _clip(hs * 1.4), _short(best)


def _cut_points(ep: Episode) -> tuple[list[float], list[float]]:
    """Starts/ends of camera shots: editors cut on shot boundaries, never mid-line."""
    shots = [e for e in ep.events if e.kind == "CAMERA"] or ep.events
    dlg = [e for e in ep.events if e.kind == "DIALOGUE"]

    def splits_line(t: float) -> bool:
        return any(d.start < t - 0.3 and d.end > t + 0.3 for d in dlg)

    starts = sorted({round(s.start, 2) for s in shots if not splits_line(s.start)})
    ends = sorted({round(s.end, 2) for s in shots if not splits_line(s.end)})
    return starts, ends


def _story(ep: Episode, a: float, b: float) -> list[Event]:
    return [e for e in ep.story_events if e.start >= a - 0.05 and e.start < b]


def _short(e: Event, n: int = 110) -> str:
    t = e.label()
    return t if len(t) <= n else t[: n - 1] + "…"


STOP = {"a", "an", "the", "he", "she", "was", "is", "i", "my", "of", "and", "to", "in", "her", "his"}
PREMISE_TERMS: set[str] = set()


def set_premise(title: str) -> None:
    """Words from the show title (e.g. married / broken / soldier / duke) are the premise a
    stranger must grasp; clips that say them out loud explain themselves."""
    PREMISE_TERMS.clear()
    for w in re.findall(r"[a-z']+", title.lower()):
        if w not in STOP and len(w) > 3:
            PREMISE_TERMS.add(w[:5])   # crude stem: married/marry -> "marri"/"marry"


def score_window(ep: Episode, a: float, b: float, next_ep: Episode | None = None) -> Candidate:
    seg = Segment(ep.number, a, b)
    c = Candidate([seg])
    inside = _story(ep, a, b)
    reasons: list[str] = []

    # ---- hook: what the viewer sees/hears in the first 3 seconds
    first_line = next((e for e in inside if e.kind == "DIALOGUE"), None)
    if cfg().audible_hook:
        c.hook_score, c.hook = _hook_v2(ep, a, reasons)
    else:
        _hook_v1(ep, a, c, reasons)
    _rest_of_window(ep, a, b, next_ep, c, seg, inside, first_line, reasons)
    return c


def _hook_v1(ep: Episode, a: float, c: "Candidate", reasons: list[str]) -> None:
    head = [e for e in ep.events if a - 0.05 <= e.start < a + HOOK_SECS]
    head_story = [e for e in head if e.kind != "CAMERA"]
    hook_beats = sorted(head, key=beat_score, reverse=True)
    best = hook_beats[0] if hook_beats else None
    hs = beat_score(best) if best else 0.0
    hs += 0.3 * sum(max(0.0, beat_score(e)) for e in head_story[1:3])
    if not head_story:
        hs -= 1.5; reasons.append("hook: no line/action in first 3s")
    c.hook_score = _clip(hs * 1.4)
    if best:
        c.hook = _short(max(head_story or head, key=beat_score))
        reasons += ["hook " + r for r in beat_reasons(best)]


def _rest_of_window(ep, a, b, next_ep, c, seg, inside, first_line, reasons) -> None:
    # ---- cliffhanger: the last seconds + what we withhold
    tail = [e for e in ep.story_events if e.end <= b + 0.05 and e.end > b - CLIFF_SECS]
    after = [e for e in ep.story_events if b - 0.05 <= e.start < b + 6]
    is_end = b >= ep.duration - 1.5
    if is_end and next_ep is not None:
        after = after + [e for e in next_ep.story_events if e.start < 6]
    cs, cwhy = cliff_signals(tail, after, is_end, calm_fix=cfg().calm_fix)
    c.cliff_score = _clip(cs * 1.6)
    reasons += ["cliff: " + w for w in cwhy]
    last_dlg = [e for e in tail if e.kind != "CAMERA"]
    if last_dlg:
        c.cliffhanger = _short(last_dlg[-1])

    # ---- standalone: will a stranger follow it?
    st = 7.0
    if first_line and REPLY_START.match(first_line.text) and first_line.start - a < 6:
        st -= 2; reasons.append("standalone: opens mid-conversation")
    prev = [e for e in ep.events if e.kind == "DIALOGUE" and a - 2.0 <= e.end <= a + 0.3]
    if prev and prev[-1].text.rstrip().endswith("?"):
        st -= 1.5; reasons.append("standalone: cuts off the question being answered")
    speakers = {e.speaker for e in inside if e.kind == "DIALOGUE"}
    if len(speakers) > 4:
        st -= 0.7 * (len(speakers) - 4); reasons.append(f"standalone: {len(speakers)} speakers")
    scenes = {e.scene for e in inside}
    if len(scenes) > 1:
        st -= 1.5 * (len(scenes) - 1); reasons.append("standalone: crosses a scene cut")
    n_lines = sum(1 for e in inside if e.kind == "DIALOGUE")
    if n_lines < 3:
        st -= 2; reasons.append("standalone: too little dialogue")
    if 28 <= seg.length <= 50:
        st += 1.0
    if PREMISE_TERMS:
        said = {w[:5] for e in inside if e.kind == "DIALOGUE" for w in re.findall(r"[a-z']+", e.text.lower())}
        hits = PREMISE_TERMS & said
        if hits:
            st += 1.5; reasons.append("standalone: says the premise (" + ", ".join(sorted(hits)) + ")")

    # dead air: long stretches with no line/action lose a scrolling viewer
    marks = [a] + sorted(e.start for e in inside) + [b]
    gaps = sum(1 for x, y in zip(marks, marks[1:]) if y - x > 6.0)
    if gaps:
        st -= 1.0 * gaps; reasons.append(f"standalone: {gaps} dead-air gap(s) >6s")
    c.standalone_score = _clip(st)

    # ---- intensity: is the middle dramatic too?
    vals = [max(0.0, beat_score(e)) for e in inside]
    c.density_score = _clip(2.2 * (sum(vals) / max(1, len(vals))))

    c.reasons = reasons
    c.total = weighted(c)
    apply_gates(c)


def episode_candidates(ep: Episode, next_ep: Episode | None = None, keep: int = 3) -> list[Candidate]:
    starts, ends = _cut_points(ep)
    pool: list[Candidate] = []
    for a in starts:
        for b in ends:
            if MIN_LEN <= b - a <= MAX_LEN:
                pool.append(score_window(ep, a, b, next_ep))
    pool.sort(key=lambda c: (c.eligible, c.total), reverse=True)   # v1: all eligible -> total only
    picked = nms(pool, keep)
    if cfg().flash_forward:
        picked = [maybe_flash_forward(ep, c, next_ep) for c in picked]
    for i, c in enumerate(picked, 1):
        c.id = f"E{ep.number:03d}-{i}"
    return picked


def overlap(c1: Candidate, c2: Candidate) -> float:
    inter = 0.0
    for s1 in c1.segments:
        for s2 in c2.segments:
            if s1.episode == s2.episode:
                inter += max(0.0, min(s1.end, s2.end) - max(s1.start, s2.start))
    return inter / max(1e-6, min(c1.length, c2.length))


def nms(pool: list[Candidate], keep: int, max_overlap: float = 0.3) -> list[Candidate]:
    out: list[Candidate] = []
    for c in pool:
        if all(overlap(c, o) <= max_overlap for o in out):
            out.append(c)
        if len(out) >= keep:
            break
    return out


def maybe_flash_forward(ep: Episode, c: Candidate, next_ep: Episode | None) -> Candidate:
    """Trailer trick: if the opening is weak, open on a 2-5s flash of the withheld payoff
    (a strong beat after the cut, same episode or start of the next), then play the build-up.
    Still ends on the same cliffhanger, total stays <= 60s."""
    if c.hook_score >= 6.0:
        return c
    body = c.segments[0]
    sources = [(ep, [e for e in ep.events if e.start >= body.end])]
    if body.end >= ep.duration - 1.5 and next_ep is not None:
        sources.append((next_ep, [e for e in next_ep.events if e.start < 30]))
    best = None
    for src_ep, evs in sources:
        for e in evs:
            if e.kind == "CAMERA":
                continue
            sc = beat_score(e)
            if sc >= 4 and (best is None or sc > best[1]):
                best = (src_ep, sc, e)
    if best is None:
        return c
    src_ep, sc, e = best
    s_starts, s_ends = _cut_points(src_ep)
    a = max([x for x in s_starts if x <= e.start + 0.05] or [e.start])
    ends = [x for x in s_ends if a + 2.0 <= x <= a + 6.0]
    if not ends:
        return c
    b = min([x for x in ends if x >= e.end - 0.3] or [max(ends)])
    flash = Segment(src_ep.number, a, b)
    if c.length + flash.length > MAX_LEN:
        # trim the body's start (to a real cut point) so the total stays <= 60s
        b_starts, _ = _cut_points(ep)
        ok = [x for x in b_starts if x >= body.end - (MAX_LEN - flash.length) and x >= body.start
              and body.end - x >= MIN_LEN - flash.length]
        if not ok:
            return c
        body = Segment(body.episode, min(ok), body.end)
    if cfg().rescore_assembled:
        return _assembled_v2(ep, src_ep, flash, body, c, next_ep)
    new = Candidate([flash, body], hook_score=_clip(sc * 1.4), cliff_score=c.cliff_score,
                    standalone_score=_clip(c.standalone_score - 0.5), density_score=c.density_score,
                    cliffhanger=c.cliffhanger, reasons=c.reasons + [
                        f"multi: opens on flash-forward from ep {src_ep.number} @ {fmt(a)} (original hook weak)"])
    new.hook = _short(e)
    new.total = weighted(new)
    apply_gates(new)
    return new if new.total > c.total else c


def _assembled_v2(ep: Episode, src_ep: Episode, flash: Segment, body: Segment, orig: "Candidate",
                  next_ep: Episode | None) -> "Candidate":
    """v2: score the promo the viewer actually sees. Hook = first 3s of the flash; cliffhanger =
    end of the (possibly trimmed) body; standalone = body's, minus 1 for the time jump;
    intensity over everything shown. The flash shows a moment from AFTER the cut, so we flag
    that it may pre-reveal the payoff instead of pretending it is free."""
    body_c = score_window(ep, body.start, body.end, next_ep)
    reasons: list[str] = []
    hook_score, hook_txt = _hook_v2(src_ep, flash.start, reasons)
    new = Candidate([flash, body])
    new.hook_score, new.hook = hook_score, hook_txt
    new.cliff_score, new.cliffhanger = body_c.cliff_score, body_c.cliffhanger
    new.standalone_score = _clip(body_c.standalone_score - 1.0)
    shown = _story(src_ep, flash.start, flash.end) + _story(ep, body.start, body.end)
    vals = [max(0.0, beat_score(x)) for x in shown]
    new.density_score = _clip(2.2 * (sum(vals) / max(1, len(vals))))
    new.reasons = ([r for r in reasons] + [r for r in body_c.reasons if not r.startswith("hook")]
                   + [f"multi: opens on flash-forward from ep {src_ep.number} @ {fmt(flash.start)}",
                      "standalone: -1 for the time jump"])
    new.flags = ["flash shows a moment from after the cut: check it doesn't give away the payoff"]
    if src_ep.number != ep.number:
        new.flags.append(f"flash taken from episode {src_ep.number}: spoiler risk")
    new.total = weighted(new)
    apply_gates(new)
    # keep the assembled version only if it is eligible-better
    return new if (new.eligible, new.total) > (orig.eligible, orig.total) else orig


def transcript(episodes: dict[int, Episode], c: Candidate, include_camera: bool = False) -> str:
    out = []
    for s in c.segments:
        ep = episodes[s.episode]
        out.append(f"--- Ep {s.episode} {fmt(s.start)}-{fmt(s.end)} ---")
        for e in ep.events:
            if s.start - 0.05 <= e.start < s.end and (include_camera or e.kind != "CAMERA"):
                out.append(f"[{fmt(e.start)}] {e.label()}")
    return "\n".join(out)
