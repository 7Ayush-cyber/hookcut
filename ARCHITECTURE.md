# HookCut architecture

```
episode .txt files
  │
  ├─ parser.py ─────────── screenplay → events (kind, start, end, speaker, tone, text, scene)
  ├─ integrity.py ──────── parse_report.json: blocks vs parsed, bad timestamps, duplicate episode
  │                        numbers, instruction-like text   (nothing fails silently)
  ├─ candidates.py ─────── every 20–60s window cut on shot boundaries, never mid-line (~250 per episode)
  │    └ heuristics.py     rule scores, 0–10: hook (audible beat in 0–3s) · cliffhanger (last 5s +
  │                        withheld payoff) · standalone · intensity
  │    └ scoring.py        versioned config (v1 original, v2, v3 default): weights, gates, switches
  │    └ gates (A)         hook ≥ 4, cliffhanger ≥ 3, standalone ≥ 4, audible beat, not resolved
  │                        → failing promos are kept but marked "not post-ready"
  │    └ per episode       top 3 by (eligible, score), overlap ≤ 30%
  ├─ pipeline.py
  │    └ [optional] LLM judge, 1 call/episode: fenced screenplay, scores clamped 0–10, quotes
  │                 fuzzy-verified, new times must be real cut points, injection → verdict ignored
  │    └ portfolio  post-first: eligible only, one per episode, overlap ≤ 10%, ≥1 promo from eps 1–3
  │    └ outputs    promos.json · promos.csv · report.md (with transcripts) · parse_report.json
  ├─ llm.py ────────────── one complete(prompt) → str; gemini | openai-compatible | anthropic | mock
  └─ app.py ────────────── Streamlit: run, review, gates/flags, editor decisions → CSV
```

**Where AI is used, and where it isn't**
- **Code:** everything that must be exact or reproducible (parsing, times, lengths, overlaps, gates, ranking).
- **Optional LLM:** the judgements rules can't make (needs context? truly unresolved?). It is never allowed to emit a time or a quote that the script doesn't contain.
- **Human:** the editor watches the shortlisted clips and records accept/revise/reject.

**Cost and latency**
- **Rules only:** about 4–5s for 50 episodes on a laptop, $0.
- **LLM judge (estimated from prompt sizes, not measured):** about 70k input + 35k output tokens per 50-episode show, i.e. 50 calls rate-limited for free tiers (about 6 min).

**Reproducibility:** rules-only runs are deterministic. `--scoring v1` reproduces the original submission exactly.

**Evaluation tooling** (`eval/`)
- `analysis.py`: distributions, correlations, effective weights, 300-draw sensitivity, ablation, gate-threshold sweeps
- `make_annotation_sheet.py`: blind, shuffled sheet with a hidden key
- `score_annotations.py`, `compare_versions.py`, `gate_agreement.py`
- `tests/test_adversarial.py`: red-team cases plus invariants on the real pack
