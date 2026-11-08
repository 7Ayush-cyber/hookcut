"""Build a BLIND annotation pool: promos from v1 and v2 post-first lists plus random samples of
v2-eligible and v2-gate-rejected promos. Shuffled, anonymised, no scores shown.
usage: python eval/make_annotation_sheet.py v1/promos.json v2/promos.json "<show folder>" out_dir
"""
import csv, json, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from promo_finder.candidates import Candidate, Segment, transcript  # noqa
from promo_finder.parser import load_show, to_sec  # noqa

v1, v2, folder, out = sys.argv[1:5]
out = Path(out); out.mkdir(parents=True, exist_ok=True)
eps = {e.number: e for e in load_show(folder)}
A, B = json.load(open(v1)), json.load(open(v2))
key = lambda p: tuple((s["episode"], s["start"], s["end"]) for s in p["segments"])
pool, origin = {}, {}
for tag, lst in (("v1_top10", A["post_first"]), ("v2_top10", B["post_first"])):
    for p in lst:
        pool[key(p)] = p; origin.setdefault(key(p), []).append(tag)
rng = random.Random(42)
rest = [p for p in B["all_promos_by_episode"] if key(p) not in pool]
for tag, cond, k in (("v2_eligible_random", True, 8), ("v2_gate_rejected_random", False, 8)):
    for p in rng.sample([p for p in rest if p["eligible"] == cond and key(p) not in pool], k):
        pool[key(p)] = p; origin.setdefault(key(p), []).append(tag)
items = list(pool.items()); rng.shuffle(items)
rows, sheet = [], ["# Blind promo annotation sheet\n",
                   "Judge each clip ONLY from the transcript (what the viewer sees/hears). Scores are hidden.\n"
                   "Use the rubric in HUMAN_EVAL_RUBRIC.md. Fill annotations_blank.csv.\n"]
for i, (k, p) in enumerate(items, 1):
    code = f"P{i:02d}"
    c = Candidate([Segment(e, to_sec(a), to_sec(b)) for e, a, b in k])
    sheet.append(f"\n## {code} — {round(c.length)}s, {len(k)} segment(s)\n```\n{transcript(eps, c)}\n```")
    rows.append({"code": code, "segments": " + ".join(f"ep{e} {a}-{b}" for e, a, b in k),
                 "origin": "|".join(origin[k]), "eligible_v2": p.get("eligible"),
                 "gate_fail_v2": "; ".join(p.get("gate_fail", []))})
(out / "annotation_sheet_blind.md").write_text("\n".join(sheet), encoding="utf-8")
with open(out / "annotation_key_DO_NOT_OPEN_BEFORE_LABELLING.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
with open(out / "annotations_blank.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["code", "hook_1to5", "cliff_1to5", "standalone_1to5", "verdict(accept/revise/reject)", "note"])
    for r in rows:
        w.writerow([r["code"], "", "", "", "", ""])
print(len(rows), "items")
