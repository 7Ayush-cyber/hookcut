"""Rule-based beat scoring. Cheap, explainable, runs on every possible window.

The LLM (optional) only re-judges the shortlist this produces.
All word lists are deliberately small and editable: tune them per genre.
"""
from __future__ import annotations

import re

from .parser import Event

HIGH_TONES = {
    "screaming", "shouting", "furious", "outraged", "horrified", "gasping", "sobbing",
    "menacing", "threatening", "venomous", "sinister", "chilling", "shocked", "panicked",
    "desperate", "weeping", "crying", "enraged", "terrified", "roaring", "hysterical",
    "explosive", "crying out", "bellowing", "frantic", "devastated", "piercing",
}
MID_TONES = {
    "cold", "mocking", "cruel", "smug", "defiant", "taunting", "scornful", "accusatory",
    "accusing", "spiteful", "commanding", "stern", "fierce", "harsh", "condescending",
    "arrogant", "dismissive", "steely", "sharp", "demanding", "bitter", "gloating",
    "incredulous", "heartbroken", "trembling", "pleading", "urgent", "final", "unyielding",
    "callous", "contemptuous", "vicious", "viciously calm", "dangerous", "lethal",
}
CALM_TONES = {
    "gentle", "soft", "tender", "reassuring", "calm", "warm", "sincere", "earnest",
    "affectionate", "loving", "relieved", "content", "playful", "amused", "fond",
}

PHYSICAL = re.compile(
    r"\b(slap\w*|punch\w*|strik\w+|struck|seiz\w+|grab\w*|drag\w*|shov\w+|kneel\w*|one knee|"
    r"kiss\w*|throw\w*|threw|hurl\w*|crash\w*|collaps\w+|faint\w*|blood\w*|sword|knife|dagger|"
    r"blade|chains?|handcuff\w*|shackl\w+|fire|burn\w*|rips?|ripp\w+|chok\w+|pistol|gun\w*|"
    r"whip\w*|scar\w*|tackl\w+|smash\w*|shatter\w*|stab\w*|wound\w*|pins? \w+ against|"
    r"lunges?|bursts?|storms? in|kicks?|kicked|spits?|falls? to)\b", re.I)
STAKES = re.compile(
    r"\b(duke|duchess|secret|truth|pregnan\w+|dead|death|die|alive|heir|betray\w*|marry|married|"
    r"marriage|wedding|divorce|disown\w*|arrest\w*|prison|kill\w*|murder\w*|poison\w*|liar|lie[sd]?|"
    r"who are you|never|wife|husband|inheritance|bastard|traitor|guards|exile\w*|ruin\w*|"
    r"revenge|hang\w*|confess\w*|identity|title|crown|king|queen|baby|child)\b", re.I)
SFX_HIT = re.compile(r"\b(slap|punch|gasp\w*|scream\w*|crash\w*|thunder\w*|shatter\w*|gunshot|"
                     r"thud|bang\w*|smash\w*|clang\w*|crack\w*|sting|stab|whoosh|slam\w*)\b", re.I)
CLIFF_SFX = re.compile(r"\b(cliffhanger|ominous|drone|sting|tension|suspense\w*|swell\w*|"
                       r"silence|cut to black|dread|heartbeat|ticking)\b", re.I)
ENTRANCE = re.compile(r"\b(enters?|bursts? (in|open|through)|doors? (open|burst|fly|swing|bang)\w*|"
                      r"arriv\w+|appears?|steps? (in|forward|out)|footsteps|o\.s\.|voice from|"
                      r"stop!|halt|reveal\w*|turns? to face)\b", re.I)
THREAT = re.compile(r"\b(will|or else|regret|pay for|dare|until|unless|going to|watch|"
                    r"never|ruin|destroy|end you|last time|warn\w*)\b", re.I)
RESOLVE = re.compile(r"\b(smil\w+ warmly|embrac\w+|hugs?|thank\w*|fades? out|peace\w*|gently|"
                     r"relie\w+|laugh\w+ together|settles?|safe now|it's over|goodnight|"
                     r"rests?|calm\w*|tender\w*|soft\w*)\b", re.I)
REPLY_START = re.compile(r"^(and|but|so|yes|no|then|because|well|or|also|too|that|it|this|he|she|they)\b", re.I)


def _tones(e: Event) -> list[str]:
    return [t.strip().lower() for t in e.tone.split(",") if t.strip()]


def beat_score(e: Event) -> float:
    """How loud / dramatic is one event, 0..~8."""
    s = 0.0
    txt = e.text
    w = 0.5 if e.kind == "CAMERA" else 1.0   # camera notes often restate the action
    for t in _tones(e):
        if t in HIGH_TONES:
            s += 2.0
        elif t in MID_TONES:
            s += 1.0
        elif t in CALM_TONES:
            s -= 1.0
    s += w * 2.0 * min(2, len(PHYSICAL.findall(txt)))
    s += w * 1.0 * min(2, len(STAKES.findall(txt)))
    if e.kind == "SFX":
        s += 2.0 * min(1, len(SFX_HIT.findall(txt)))
    if e.kind == "DIALOGUE":
        if "!" in txt:
            s += 0.7
        if "?" in txt:
            s += 0.4
        words = len(txt.split())
        if words <= 7 and ("!" in txt or "?" in txt):
            s += 0.5
    return max(-2.0, s)


def beat_reasons(e: Event) -> list[str]:
    r = []
    hi = [t for t in _tones(e) if t in HIGH_TONES]
    if hi:
        r.append("tone:" + "/".join(hi))
    r += ["action:" + m.lower() for m in PHYSICAL.findall(e.text)[:2]]
    r += ["stakes:" + m.lower() for m in STAKES.findall(e.text)[:2]]
    if e.kind == "SFX" and SFX_HIT.search(e.text):
        r.append("sfx-hit")
    return r


def cliff_signals(last: list[Event], after: list[Event], is_episode_end: bool,
                  calm_fix: bool = False) -> tuple[float, list[str]]:
    """Score how unresolved the cut point is.

    last  = story events that finish in the final seconds of the clip
    after = story events right after the cut (what the viewer *doesn't* see)
    """
    s, why = 0.0, []
    text = " ".join(e.text for e in last)
    dlg = [e for e in last if e.kind == "DIALOGUE"]
    calm_last = bool(dlg) and any(t in CALM_TONES for t in _tones(dlg[-1]))
    if dlg and not (calm_fix and calm_last):   # v2: a tender question is not a cliffhanger
        line = dlg[-1].text
        if line.rstrip().endswith("?"):
            s += 1.5; why.append("ends on a question")
        if line.rstrip().endswith(("...", "…", "—", "-")):
            s += 1.0; why.append("line cut off")
        if THREAT.search(line):
            s += 1.0; why.append("threat/promise")
        if any(t in HIGH_TONES for t in _tones(dlg[-1])):
            s += 1.0; why.append("high-emotion last line")
    if ENTRANCE.search(text):
        s += 1.5; why.append("entrance/reveal")
    if any(e.kind == "SFX" and CLIFF_SFX.search(e.text) for e in last):
        s += 2.0; why.append("scored as a sting")
    if PHYSICAL.search(text):
        s += 1.0; why.append("mid-action")
    if is_episode_end:
        s += 1.0; why.append("writer's own episode cliffhanger")
    # the payoff we are withholding: strong beats right after the cut = good cut point
    withheld = max([beat_score(e) for e in after] or [0.0])
    if withheld >= 3:
        s += min(2.0, withheld / 3); why.append("cuts right before a big beat")
    if RESOLVE.search(text):
        s -= 1.5; why.append("PENALTY: feels resolved")
    if calm_last:
        s -= 1.0; why.append("PENALTY: calm last line")
    return s, why
