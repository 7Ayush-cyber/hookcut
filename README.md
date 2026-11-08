# HookCut: ranked promo clips from microdrama screenplays

Kuku AI Generalist assignment, Problem 1 (Find the Promos). By **Ayush Kumar** (230104021, Civil Engineering, IIT Guwahati).

The solution document (2-page PDF), the raw outputs on the 50-episode pack and the evaluation files are in the submission Drive folder:
https://drive.google.com/drive/folders/1VXz1SvRiXrtnzEH6NZ44ySXPCykKNvCe

The input screenplays are Kuku's confidential content, so they are **not** in this repo. Point `--input` at any folder of episode `.txt` files in the same format. `tests/fixtures/` has a small synthetic sample you can try:

```bash
pip install -r requirements.txt
python run.py --input tests/fixtures --out outputs/sample
```

Reads a show's timestamped screenplays (one `.txt` per episode) and returns 20–60s promo clips.
Each one comes with **episode + start/end of every segment, the hook, the cliffhanger, one line on why it works**,
and a ranked **"post these first"** list. An editor reviews the shortlist; nothing is auto-posted.

```
screenplays ──► parser (code) ──► every 20–60s window cut on shot boundaries (code)
           ──► rule scores: hook(first 3s) · cliffhanger(last 5s + withheld payoff) · standalone · intensity
           ──► top 3 per episode (non-overlapping) + optional flash-forward opener
           ──► [optional] LLM judge re-scores, rewrites hook/why, may nudge cuts
                 └─ guardrails: times must be real cut points, quotes must exist in the script
           ──► rank across the show ──► promos.json / promos.csv / report.md
```

## Run it

```bash
pip install -r requirements.txt

# 1) Free, offline, ~5 s for 50 episodes
python run.py --input "path/to/Show Title" --out outputs/show

# 2) With an LLM judge (one call per episode, ~70k input tokens per 50-episode show)
export PROMO_LLM_API_KEY=...                      # free key: https://aistudio.google.com/apikey
python run.py --input "path/to/Show Title" --out outputs/show_llm --provider gemini

# any OpenAI-compatible endpoint: Groq, OpenRouter, Ollama, an in-house gateway
python run.py --input ... --provider openai --base-url https://api.groq.com/openai/v1 --model llama-3.3-70b-versatile
python run.py --input ... --provider anthropic --model claude-sonnet-4-5

# UI
streamlit run app.py          # upload the .txt files or a .zip, pick a model in the sidebar
```

The folder name is used as the show title, so its words (e.g. *married, soldier, duke*) count as the
premise a stranger must understand. You can override it with `--title`.

## Scoring versions
`--scoring v3` is the default. `v1` reproduces the original submission exactly, and `v2` is the first set of audit fixes.

v3 adds non-compensatory quality gates, an audible-hook rule, a calm-question fix, and LLM prompt-injection guards. Flash-forward multi-segment promos are off by default, because the two AI reviewers disagreed on them (0/7 vs 3/7 accepted). Turn them on with `--flash-forward`.

See `ARCHITECTURE.md`, and the audit report in the submission folder.

## Swapping the model
All model calls go through `promo_finder/llm.py → LLM.complete(prompt) -> str` (plain `requests`, no SDKs).
Set `PROMO_LLM_PROVIDER / PROMO_LLM_MODEL / PROMO_LLM_API_KEY / PROMO_LLM_BASE_URL`, or use the CLI flags.
A new provider takes about 15 lines.

## Tests
- `python tests/test_adversarial.py` runs the red-team cases (injection, resolved endings, dead openings, malformed timestamps, duplicate episodes…), plus invariants on the real pack when it's present.
- `python tests/test_pipeline.py` These check invariants on a synthetic sample: every clip is 20–60s, every timestamp
exists in the script, no clip cuts through a line of dialogue, the post-first list has one promo per episode,
and the LLM path rejects quotes that aren't in the script. Point `PROMO_TEST_SHOW` at a real show folder to run them on it.

## Files
| file | what |
|---|---|
| `promo_finder/parser.py` | screenplay → events (dialogue/action/camera/SFX with times, speaker, tone) |
| `promo_finder/heuristics.py` | editable word lists and beat / cliffhanger scoring |
| `promo_finder/candidates.py` | window generation, scoring, de-duplication, flash-forward openers |
| `promo_finder/pipeline.py` | LLM judge prompt and validation, ranking, outputs |
| `promo_finder/llm.py` | swappable providers (gemini / openai-compatible / anthropic / mock) |

Screenplays and outputs are git-ignored: the input pack is confidential.
