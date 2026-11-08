"""Before/after table: for each version's post-first list, look up reviewer labels by exact segments.
usage: python eval/compare_versions.py labels.csv key.csv addendum.csv v1.json v2.json v3.json"""
import csv, json, sys
labels, key, add = sys.argv[1:4]
L = {r["code"]: r for r in csv.DictReader(open(labels))}
by_seg = {r["segments"]: L[r["code"]] for r in csv.DictReader(open(key)) if r["code"] in L}
for r in csv.DictReader(open(add)):
    by_seg[r["segments"]] = r
out = {}
for path in sys.argv[4:]:
    d = json.load(open(path)); name = d["meta"].get("scoring", path)
    rows = []
    for p in d["post_first"]:
        seg = " + ".join(f"ep{s['episode']} {s['start']}-{s['end']}" for s in p["segments"])
        lab = by_seg.get(seg)
        rows.append({"id": p["id"], "segments": seg, "type": p["type"],
                     "verdict": lab["verdict"] if lab else "UNLABELLED",
                     "non_blind": bool(lab and "NON-BLIND" in lab.get("note", ""))})
    v = [r["verdict"] for r in rows]
    eps = sorted(int(r["segments"].split()[0][2:]) for r in rows)
    out[name] = {"accept": v.count("accept"), "revise": v.count("revise"), "reject": v.count("reject"),
                 "unlabelled": v.count("UNLABELLED"), "non_blind_labels": sum(r["non_blind"] for r in rows),
                 "multi_segment": sum(r["type"] != "continuous" for r in rows),
                 "adjacent_episode_pairs": sum(1 for i, a in enumerate(eps) for b in eps[i + 1:] if abs(a - b) == 1),
                 "eligible_in_all_promos": d["meta"].get("eligible_candidates"), "rows": rows}
print(json.dumps(out, indent=2))
