"""Scoring-system audit: correlations, effective weights, compensation, sensitivity, ablation,
gate-threshold sensitivity. Runs on the real pack, rules only, continuous windows only
(flash-forward assembly is excluded here to keep 300+ re-rankings fast; noted in the report).

usage: python eval/analysis.py "<show folder>" eval_out/
"""
import json
import random
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from promo_finder.candidates import nms, score_window, set_premise, _cut_points, MIN_LEN, MAX_LEN  # noqa
from promo_finder.parser import load_show  # noqa
from promo_finder.scoring import V1, V2, V3, set_config  # noqa

DIMS = ["hook", "cliff", "standalone", "intensity"]


def precompute(eps, conf):
    set_config(conf)
    out = {}
    for n, ep in eps.items():
        starts, ends = _cut_points(ep)
        rows = []
        for a in starts:
            for b in ends:
                if MIN_LEN <= b - a <= MAX_LEN:
                    c = score_window(ep, a, b, eps.get(n + 1))
                    rows.append(c)
        out[n] = rows
    return out


def comp(c):
    return {"hook": c.hook_score, "cliff": c.cliff_score, "standalone": c.standalone_score,
            "intensity": c.density_score}


def rank(pre, weights, arc, conf, gates=None, gap=1, premise=False, top=10):
    n_eps = len(pre)
    picked = []
    for n, rows in pre.items():
        def tot(c):
            v = comp(c)
            return sum(weights[k] * v[k] for k in DIMS)

        def elig(c):
            if not gates:
                return True
            v = comp(c)
            if any(r.startswith("hook: no audible") for r in c.reasons) or any("feels resolved" in r for r in c.reasons):
                return False
            return v["hook"] >= gates["hook"] and v["cliff"] >= gates["cliff"] and v["standalone"] >= gates["standalone"]
        scored = sorted(rows, key=lambda c: (elig(c), tot(c)), reverse=True)
        for c in scored:
            c.total = tot(c)
        for c in nms(scored, 3):
            picked.append((n, c.segments[0].start, c.segments[0].end, tot(c) + arc * (1 - (n - 1) / (n_eps - 1)), elig(c)))
    picked.sort(key=lambda r: r[3], reverse=True)
    pool = [p for p in picked if p[4]] if gates else picked
    out = []
    for p in pool:
        if any(abs(p[0] - o[0]) < max(1, gap) for o in out):
            continue
        out.append(p)
        if len(out) >= top:
            break
    if premise and out and not any(o[0] <= 3 for o in out):
        early = [p for p in pool if p[0] <= 3]
        if early:
            out[-1] = early[0]
    return [(p[0], round(p[1], 2), round(p[2], 2)) for p in out], picked


def jacc(a, b):
    a, b = set(x[0] for x in a), set(x[0] for x in b)     # compare by episode (one promo per ep)
    return len(a & b) / max(1, len(a | b))


def spearman(x, y):
    def rk(v):
        o = sorted(range(len(v)), key=lambda i: v[i]); r = [0] * len(v)
        for i, j in enumerate(o):
            r[j] = i
        return r
    rx, ry = rk(x), rk(y)
    mx, my = st.mean(rx), st.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else 0.0


def main(folder, out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    set_premise(Path(folder).name)
    eps = {e.number: e for e in load_show(folder)}
    report = {}
    for conf in (V1, V2, V3):
        pre = precompute(eps, conf)
        allc = [c for rows in pre.values() for c in rows]
        vals = {k: [comp(c)[k] for c in allc] for k in DIMS}
        r = {"windows": len(allc)}
        r["distribution"] = {k: {"mean": round(st.mean(v), 2), "sd": round(st.pstdev(v), 2),
                                 "pct_at_10": round(100 * sum(x >= 9.99 for x in v) / len(v), 1),
                                 "pct_at_0": round(100 * sum(x <= 0.01 for x in v) / len(v), 1)}
                             for k, v in vals.items()}
        r["spearman"] = {f"{a}~{b}": round(spearman(vals[a], vals[b]), 2)
                         for i, a in enumerate(DIMS) for b in DIMS[i + 1:]}
        w = conf.weights
        sd = {k: st.pstdev(vals[k]) for k in DIMS}
        tot = sum(w[k] * sd[k] for k in DIMS)
        r["nominal_weight"] = w
        r["effective_weight(w*sd share)"] = {k: round(w[k] * sd[k] / tot, 3) for k in DIMS}
        r["length_vs_total_spearman"] = round(spearman([c.segments[0].length for c in allc],
                                                       [sum(w[k] * comp(c)[k] for k in DIMS) for c in allc]), 2)
        gates = conf.gates
        base, picked = rank(pre, w, conf.arc_max, conf, gates, conf.min_episode_gap, conf.premise_slot)
        r["baseline_top10"] = base
        # compensation: shortlisted promos (top-3/episode) that rank high despite a weak dimension
        thr = {"hook": 4.0, "cliff": 3.0, "standalone": 4.0}
        top30 = sorted(picked, key=lambda p: p[3], reverse=True)[:30]
        def find(p):
            for c in pre[p[0]]:
                if round(c.segments[0].start, 2) == round(p[1], 2) and round(c.segments[0].end, 2) == round(p[2], 2):
                    return comp(c)
        weak = [(p[0], {k: round(v, 1) for k, v in find(p).items() if k in thr and v < thr[k]}) for p in top30]
        r["top30_with_a_weak_dimension"] = [x for x in weak if x[1]]
        # sensitivity: Dirichlet perturbation of weights, plus arc variation
        rng = random.Random(0)
        js, keep = [], {e[0]: 0 for e in base}
        for _ in range(300):
            g = {k: rng.gammavariate(40 * w[k], 1) for k in DIMS}
            s_ = sum(g.values()); ww = {k: g[k] / s_ for k in DIMS}
            arc = rng.choice([0.0, 0.3, 0.6, 0.9])
            t, _ = rank(pre, ww, arc, conf, gates, conf.min_episode_gap, conf.premise_slot)
            js.append(jacc(base, t))
            for e in t:
                if e[0] in keep:
                    keep[e[0]] += 1
        r["sensitivity"] = {"draws": 300, "mean_top10_overlap": round(st.mean(js), 2),
                            "min_overlap": round(min(js), 2),
                            "pct_draws_overlap>=0.7": round(100 * sum(j >= 0.7 for j in js) / 300, 1),
                            "retention_of_each_baseline_pick(ep:%)": {k: round(100 * v / 300) for k, v in keep.items()}}
        named = {"equal_weights": {k: 0.25 for k in DIMS},
                 "hook_heavy": {"hook": 0.5, "cliff": 0.25, "standalone": 0.125, "intensity": 0.125},
                 "cliff_heavy": {"hook": 0.25, "cliff": 0.5, "standalone": 0.125, "intensity": 0.125},
                 "standalone_heavy": {"hook": 0.3, "cliff": 0.3, "standalone": 0.3, "intensity": 0.1}}
        r["alternative_weights_overlap"] = {}
        for nm, ww in named.items():
            t, _ = rank(pre, ww, conf.arc_max, conf, gates, conf.min_episode_gap, conf.premise_slot)
            r["alternative_weights_overlap"][nm] = {"overlap": round(jacc(base, t), 2), "top10_eps": [x[0] for x in t]}
        # ablation: drop one dimension (renormalise), or arc
        r["ablation"] = {}
        for k in DIMS:
            ww = {d: (0 if d == k else w[d]) for d in DIMS}; s_ = sum(ww.values()); ww = {d: v / s_ for d, v in ww.items()}
            t, _ = rank(pre, ww, conf.arc_max, conf, gates, conf.min_episode_gap, conf.premise_slot)
            r["ablation"][f"drop_{k}"] = {"overlap": round(jacc(base, t), 2), "top10_eps": [x[0] for x in t]}
        t, _ = rank(pre, w, 0.0, conf, gates, conf.min_episode_gap, conf.premise_slot)
        r["ablation"]["drop_arc_bonus"] = {"overlap": round(jacc(base, t), 2), "top10_eps": [x[0] for x in t]}
        if gates:
            t, _ = rank(pre, w, conf.arc_max, conf, None, conf.min_episode_gap, conf.premise_slot)
            r["ablation"]["drop_gates"] = {"overlap": round(jacc(base, t), 2), "top10_eps": [x[0] for x in t]}
            t, _ = rank(pre, w, conf.arc_max, conf, gates, 1, conf.premise_slot)
            r["ablation"]["drop_episode_gap"] = {"overlap": round(jacc(base, t), 2), "top10_eps": [x[0] for x in t]}
            r["gate_threshold_sensitivity"] = {}
            for dim in thr:
                for d in (-1, +1):
                    g2 = dict(gates); g2[dim] = gates[dim] + d
                    t, pk = rank(pre, w, conf.arc_max, conf, g2, conf.min_episode_gap, conf.premise_slot)
                    r["gate_threshold_sensitivity"][f"{dim}{d:+d}"] = {
                        "overlap": round(jacc(base, t), 2), "eligible_shortlisted": sum(p[4] for p in pk)}
        report[conf.name] = r
        print(conf.name, "done", flush=True)
    (out / "scoring_audit.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "eval_out")
