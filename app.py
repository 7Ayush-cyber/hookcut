"""HookCut Streamlit UI: upload a show's screenplays, review ranked promos, record editor
decisions.  `streamlit run app.py`"""
import csv
import io
import json
import tempfile
import zipfile
from pathlib import Path

import streamlit as st

from promo_finder.candidates import transcript
from promo_finder.llm import DEFAULT_MODELS, LLMConfig
from promo_finder.parser import fmt
from promo_finder.pipeline import run

st.set_page_config(page_title="HookCut", layout="wide")
st.title("HookCut")
st.caption("Reads microdrama screenplays and shortlists 20–60s promo clips: hook in the first 3s, "
           "cut on a cliffhanger, understandable cold. Scores are script-based estimates, not predicted CTR. "
           "An editor decides; nothing is auto-posted.")

with st.sidebar:
    st.header("Model")
    provider = st.selectbox("LLM judge", ["none", "gemini", "openai", "anthropic"],
                            help="'none' = free rule-based scoring only. 'openai' works with any "
                                 "OpenAI-compatible endpoint (Groq, OpenRouter, Ollama, in-house).")
    model = st.text_input("Model", DEFAULT_MODELS.get(provider, ""))
    base_url = st.text_input("Base URL (openai-compatible only)", "",
                             placeholder="https://api.groq.com/openai/v1") if provider == "openai" else ""
    try:
        default_key = st.secrets.get("PROMO_LLM_API_KEY", "")
    except Exception:  # no secrets.toml
        default_key = ""
    key = "" if provider == "none" else st.text_input("API key", type="password", value=default_key)
    st.header("Scoring")
    scoring = st.selectbox("Version", ["v3", "v2", "v1"],
                           help="v3 = current (after audit + blind review); v1 = original, kept for comparison")
    st.header("Output")
    per_ep = st.slider("Promos kept per episode", 1, 5, 3)
    top_n = st.slider("'Post first' list size", 5, 20, 10)

files = st.file_uploader("Episode screenplays (.txt) or one .zip of them", type=["txt", "zip"],
                         accept_multiple_files=True)
title = st.text_input("Show title (used as the premise a stranger must understand)", "")

if st.button("Find promos", type="primary", disabled=not files):
    tmp = Path(tempfile.mkdtemp())
    show = tmp / "show"
    show.mkdir()
    zips = [f.name for f in files if f.name.endswith(".zip")]
    # the title feeds the 'premise words' signal; never fall back to the temp folder name
    title = title.strip() or (Path(zips[0]).stem if zips else "")
    if not title:
        st.warning("No show title given: the 'says the premise' signal is off for this run.")
    for f in files:
        if f.name.endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(f.read())) as z:
                for n in z.namelist():
                    if n.endswith(".txt") and not n.startswith("__MACOSX"):
                        (show / Path(n).name).write_bytes(z.read(n))
        else:
            (show / f.name).write_bytes(f.read())
    cfg = LLMConfig(provider=provider, model=model, api_key=key, base_url=base_url)
    if cfg.enabled and not key:
        st.error("Add an API key or pick 'none'.")
        st.stop()
    bar = st.progress(0.0, "Reading episodes…")
    res = run(show, tmp / "out", cfg, per_episode=per_ep, top_n=top_n, scoring=scoring,
              title=title, progress=lambda i, n, ep: bar.progress(i / n, f"Episode {ep} ({i}/{n})"))
    st.session_state["res"] = res
    st.session_state["out"] = tmp / "out"

res = st.session_state.get("res")
if res:
    m = res["meta"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Episodes", m["episodes"]); c2.metric("Promos found", m["candidates"])
    c3.metric("LLM calls", m["llm_calls"]); c4.metric("Runtime (s)", m["runtime_sec"])
    st.caption(f"Mode: {m['mode']} · scoring {m.get('scoring')} · "
               f"{m.get('eligible_candidates')} of {m['candidates']} promos pass the quality gates")
    if not m.get("parse_ok", True):
        st.warning("Parsing problems — check before trusting the results:\n\n- " + "\n- ".join(m["parse_problems"]))
    out = st.session_state["out"]
    d1, d2, d3 = st.columns(3)
    d1.download_button("promos.json", (out / "promos.json").read_bytes(), "promos.json")
    d2.download_button("promos.csv", (out / "promos.csv").read_bytes(), "promos.csv")
    d3.download_button("report.md", (out / "report.md").read_bytes(), "report.md")

    tab1, tab2, tab3 = st.tabs(["Post these first", "All promos by episode", "Editor decisions"])

    def show_card(c, label, key_prefix):
        segs = " + ".join(f"Ep {s.episode} {fmt(s.start)}–{fmt(s.end)}" for s in c.segments)
        badge = "" if c.eligible else "  ·  ⚠ not post-ready"
        with st.expander(f"{label}  ·  {segs}  ·  {c.length:.0f}s  ·  score {c.rank_score:.2f}{badge}"):
            st.markdown(f"**Hook:** {c.hook}\n\n**Cliffhanger:** {c.cliffhanger}\n\n**Why:** {c.why}")
            if c.gate_fail:
                st.warning("Fails quality gate: " + "; ".join(c.gate_fail))
            for f in c.flags:
                st.info("Editor check: " + f)
            st.json(c.to_dict()["scores"], expanded=False)
            if c.llm:
                st.caption(f"LLM: {json.dumps(c.llm)}")
            st.code(transcript(res["episodes"], c), language=None, wrap_lines=True)

    with tab1:
        for i, c in enumerate(res["post_first"], 1):
            show_card(c, f"#{i} {c.id}", "pf")
    with tab2:
        ep_sel = st.selectbox("Episode", sorted(res["episodes"]))
        for c in [c for c in res["all"] if c.episode == ep_sel]:
            show_card(c, c.id, "all")
    with tab3:
        st.write("Mark each promo after watching the clip. Export the sheet: per-show acceptance rate is the "
                 "metric that tells us whether HookCut is useful, and the labels are what the gates should be "
                 "re-tuned on.")
        pf_ids = {c.id: i for i, c in enumerate(res["post_first"], 1)}
        rows = [{"post_rank": pf_ids.get(c.id), "id": c.id,
                 "segments": " + ".join(f"ep{s.episode} {fmt(s.start)}-{fmt(s.end)}" for s in c.segments),
                 "score": round(c.rank_score, 2), "post_ready": c.eligible,
                 "decision": "", "better_in_out": "", "note": ""}
                for c in sorted(res["all"], key=lambda c: (pf_ids.get(c.id, 999), -c.rank_score))]
        edited = st.data_editor(rows, use_container_width=True, hide_index=True, key="decisions",
                                disabled=["post_rank", "id", "segments", "score", "post_ready"],
                                column_config={"decision": st.column_config.SelectboxColumn(
                                    "decision", options=["", "accept", "revise", "reject"])})
        done = [r for r in edited if r["decision"]]
        st.caption(f"{len(done)} promo(s) reviewed"
                   + (f" · accept rate {sum(r['decision'] == 'accept' for r in done) / len(done):.0%}" if done else ""))
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(edited)
        st.download_button("editor_decisions.csv", buf.getvalue(), "editor_decisions.csv")
