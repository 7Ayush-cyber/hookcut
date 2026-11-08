"""HookCut CLI: python run.py --input "path/to/show_folder" --out outputs/my_show [--provider gemini]"""
import argparse
import os

from promo_finder.llm import DEFAULT_MODELS, LLMConfig
from promo_finder.pipeline import run

ap = argparse.ArgumentParser(description="Find ranked promo clips in microdrama screenplays")
ap.add_argument("--input", required=True, help="folder of episode .txt screenplays")
ap.add_argument("--out", default="outputs/run", help="output folder")
ap.add_argument("--provider", default=os.getenv("PROMO_LLM_PROVIDER", "none"),
                help="none | gemini | openai | anthropic | mock")
ap.add_argument("--model", default=os.getenv("PROMO_LLM_MODEL", ""))
ap.add_argument("--base-url", default=os.getenv("PROMO_LLM_BASE_URL", ""))
ap.add_argument("--scoring", default="v3", choices=["v1", "v2", "v3"],
                help="v1 = original, v2 = first audit fixes, v3 = after blind review (default)")
ap.add_argument("--flash-forward", action="store_true",
                help="also build 'flash of the payoff + build-up' promos (off by default in v3; raters disagreed)")
ap.add_argument("--title", default=None, help="show title (default: folder name)")
ap.add_argument("--per-episode", type=int, default=3)
ap.add_argument("--top", type=int, default=10)
a = ap.parse_args()

os.environ["PROMO_LLM_PROVIDER"] = a.provider
cfg = LLMConfig.from_env()
cfg.model = a.model or DEFAULT_MODELS.get(a.provider, "")
cfg.base_url = a.base_url or cfg.base_url
if cfg.enabled and cfg.provider != "mock" and not cfg.api_key:
    raise SystemExit(f"No API key for {cfg.provider}: set PROMO_LLM_API_KEY")

from promo_finder.scoring import CONFIGS
scoring = CONFIGS[a.scoring].with_(flash_forward=True) if a.flash_forward else a.scoring
res = run(a.input, a.out, cfg, per_episode=a.per_episode, top_n=a.top, title=a.title, scoring=scoring,
          progress=lambda i, n, ep: print(f"\r  episode {i}/{n}", end="", flush=True))
print("\n", res["meta"])
print("\nPOST FIRST:")
for i, c in enumerate(res["post_first"], 1):
    segs = " + ".join(f"ep{s.episode} {s.to_dict()['start']}-{s.to_dict()['end']}" for s in c.segments)
    print(f"{i:>2}. {c.id}  {segs}  ({c.length:.0f}s)  score={c.rank_score:.2f}\n    HOOK: {c.hook}\n    CLIFF: {c.cliffhanger}")
print(f"\nWrote {a.out}/promos.json, promos.csv, report.md")
