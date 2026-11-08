"""Join blind annotations with the hidden key; report per-version precision and gate agreement.
usage: python eval/score_annotations.py annotations.csv annotation_key.csv [label_source]"""
import csv, json, sys
from collections import defaultdict
ann = {r["code"]: r for r in csv.DictReader(open(sys.argv[1]))}
key = {r["code"]: r for r in csv.DictReader(open(sys.argv[2]))}
src = sys.argv[3] if len(sys.argv) > 3 else "unspecified"
groups = defaultdict(list)
for code, k in key.items():
    a = ann.get(code)
    if not a or not a["verdict"]:
        continue
    for o in k["origin"].split("|"):
        groups[o].append((code, a, k))
out = {"label_source": src, "n_labelled": sum(1 for a in ann.values() if a["verdict"])}
for g, rows in sorted(groups.items()):
    v = [a["verdict"].strip().lower() for _, a, _ in rows]
    out[g] = {"n": len(rows), "accept": v.count("accept"), "revise": v.count("revise"), "reject": v.count("reject"),
              "accept_rate": round(v.count("accept") / len(v), 2),
              "usable_rate(accept+revise)": round((v.count("accept") + v.count("revise")) / len(v), 2),
              "mean_hook": round(sum(int(a["hook_1to5"]) for _, a, _ in rows) / len(rows), 2),
              "mean_cliff": round(sum(int(a["cliff_1to5"]) for _, a, _ in rows) / len(rows), 2),
              "mean_standalone": round(sum(int(a["standalone_1to5"]) for _, a, _ in rows) / len(rows), 2)}
# gate agreement on everything labelled (v2 eligibility vs reviewer verdict)
tp = fp = fn = tn = 0
for code, k in key.items():
    a = ann.get(code)
    if not a or not a["verdict"]:
        continue
    elig = k["eligible_v2"] == "True"; good = a["verdict"].strip().lower() == "accept"
    tp += elig and good; fp += elig and not good; fn += (not elig) and good; tn += (not elig) and (not good)
out["v2_gate_vs_reviewer_accept"] = {"eligible&accept": tp, "eligible&not_accept": fp,
                                     "rejected&accept(false_reject)": fn, "rejected&not_accept": tn}
print(json.dumps(out, indent=2))
