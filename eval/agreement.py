"""Inter-rater agreement between any two annotation CSVs (same codes): verdict agreement,
Cohen's kappa, and mean absolute difference on the 1-5 scales.
usage: python eval/agreement.py raterA.csv raterB.csv"""
import csv, json, sys
from collections import Counter
A = {r["code"]: r for r in csv.DictReader(open(sys.argv[1]))}
B = {r["code"]: r for r in csv.DictReader(open(sys.argv[2]))}
V = lambda r: (r.get("verdict") or r.get("verdict(accept/revise/reject)") or "").strip().lower()
codes = [c for c in A if c in B and V(A[c]) and V(B[c])]
a = [V(A[c]) for c in codes]; b = [V(B[c]) for c in codes]
po = sum(x == y for x, y in zip(a, b)) / len(codes)
ca, cb = Counter(a), Counter(b)
pe = sum(ca[k] * cb[k] for k in set(a) | set(b)) / len(codes) ** 2
out = {"n": len(codes), "verdict_agreement": round(po, 2), "cohens_kappa": round((po - pe) / (1 - pe), 2) if pe < 1 else None}
for dim in ("hook_1to5", "cliff_1to5", "standalone_1to5"):
    d = [abs(int(A[c][dim]) - int(B[c][dim])) for c in codes if A[c].get(dim) and B[c].get(dim)]
    if d:
        out[f"mean_abs_diff_{dim}"] = round(sum(d) / len(d), 2)
out["disagreements"] = [{"code": c, "A": V(A[c]), "B": V(B[c])} for c in codes if V(A[c]) != V(B[c])]
print(json.dumps(out, indent=2))
