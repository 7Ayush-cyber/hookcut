"""Versioned scoring configuration. v1 reproduces the original submission exactly;
v2 holds the changes accepted after the audit (see AUDIT_REPORT.md). Every switch can be
toggled on its own so each change can be tested in isolation (ablation)."""
from __future__ import annotations

from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class ScoringConfig:
    name: str = "v1"
    # compensatory part: weighted sum of 0-10 indices
    weights: dict = field(default_factory=lambda: {"hook": 0.35, "cliff": 0.35,
                                                   "standalone": 0.15, "intensity": 0.15})
    arc_max: float = 0.6                 # early-episode bonus added to the rank score
    # ---- v2 switches (all False in v1)
    audible_hook: bool = False           # hook must be a story beat actually heard/seen in 0-3s
    calm_fix: bool = False               # a calm/tender last line cannot earn question/threat bonus
    rescore_assembled: bool = False      # multi-segment promos re-scored as one assembled ad
    gates: dict | None = None            # non-compensatory minimums; failing = not post-able
    min_episode_gap: int = 1             # post-first: 1 = only "not the same episode" (v1)
    premise_slot: bool = False           # post-first keeps >=1 eligible promo from episodes 1-3
    llm_safety: bool = False             # clamp LLM scores, fence script text, injection flag
    flash_forward: bool = True           # auto-build "flash of the payoff + build-up" promos

    def with_(self, **kw) -> "ScoringConfig":
        return replace(self, **kw)


V1 = ScoringConfig()
V2 = ScoringConfig(
    name="v2",
    audible_hook=True,
    calm_fix=True,
    rescore_assembled=True,
    gates={"hook": 4.0, "cliff": 3.0, "standalone": 4.0},
    min_episode_gap=2,
    premise_slot=True,
    llm_safety=True,
)
# v3 = v2 after the blind review (eval/annotation_results_*.json):
#  - automatic flash-forward promos: 0/7 "accept" (Claude) vs 3/7 (Gemini) -> off by default,
#    available with run.py --flash-forward
#  - episode-gap>=2 portfolio rule pushed out 2 "accept" promos for 3 "revise" -> reverted,
#    adjacency is reported as a diagnostic instead
#  - arc bonus: removing it leaves the top-10 SET unchanged but moves the best-reviewed promo
#    (ep 3 rescue) from #7 to #10 -> kept as an ordering tie-breaker (weak evidence, see audit)
V3 = V2.with_(name="v3", flash_forward=False, min_episode_gap=1)
CONFIGS = {"v1": V1, "v2": V2, "v3": V3}

# active config (set by pipeline.run); module-level so the scoring functions stay simple
ACTIVE: list[ScoringConfig] = [V1]


def cfg() -> ScoringConfig:
    return ACTIVE[0]


def set_config(c: ScoringConfig) -> None:
    ACTIVE[0] = c
