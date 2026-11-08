"""Do the v3 gates agree with reviewer verdicts? Re-scores every labelled CONTINUOUS clip under v3
(multi-segment clips are skipped: v3 does not generate them).
usage: python eval/gate_agreement.py labels.csv key.csv addendum.csv "<show folder>" """
import csv, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from promo_finder.candidates import score_window, set_premise  # noqa
from promo_finder.parser import load_show, to_sec  # noqa
from promo_finder.scoring import CONFIGS, set_config  # noqa
labels, key, add, folder = sys.argv[1:5]
set_config(CONFIGS["v3"]); set_premise(Path(folder).name)
eps = {e.number: e for e in load_show(folder)}
L = {r["code"]: r for r in csv.DictReader(open(labels))}
items = [(r["segments"], L[r["code"]]) for r in csv.DictReader(open(key)) if r["code"] in L]
items += [(r["segments"], r) for r in csv.DictReader(open(add))]
m = {"eligible&accept": 0, "eligible&revise": 0, "eligible&reject": 0,
     "gated&accept": 0, "gated&revise": 0, "gated&reject": 0}
rows = []
for seg, lab in items:
    if "+" in seg:
        continue
    ep, rng = seg.split(" ", 1); a, b = rng.split("-")
    e = eps[int(ep[2:])]
    c = score_window(e, to_sec(a), to_sec(b), eps.get(e.number + 1))
    k = ("eligible" if c.eligible else "gated") + "&" + lab["verdict"]
    m[k] += 1
    rows.append({"segments": seg, "verdict": lab["verdict"], "v3_eligible": c.eligible, "gate_fail": c.gate_fail})
n_acc = m["eligible&accept"] + m["gated&accept"]
m["accept_recall_of_gates"] = round(m["eligible&accept"] / max(1, n_acc), 2)
m["reject_caught_by_gates"] = f'{m["gated&reject"]}/{m["gated&reject"] + m["eligible&reject"]}'
m["precision_of_eligible_for_usable(accept+revise)"] = round(
    (m["eligible&accept"] + m["eligible&revise"]) / max(1, m["eligible&accept"] + m["eligible&revise"] + m["eligible&reject"]), 2)
print(json.dumps({"summary": m, "rows": rows}, indent=2))
