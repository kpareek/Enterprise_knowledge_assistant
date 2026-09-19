"""
Evaluation dashboard -- visualizes results produced by eval/evaluate.py.

Run:
    streamlit run eval/dashboard.py
"""
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))

st.set_page_config(page_title="RAG Evaluation Dashboard", page_icon="📊", layout="wide")
st.title("📊 RAG Evaluation Dashboard")
st.caption("Retrieval relevance, answer quality, confidence calibration, and latency for the Enterprise Knowledge Assistant.")

EVAL_DIR = Path(__file__).resolve().parent
RESULTS_PATHS = {"free": EVAL_DIR / "eval_results_free.json", "paid": EVAL_DIR / "eval_results_paid.json"}


def load_results(mode: str):
    path = RESULTS_PATHS[mode]
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def to_df(data: dict) -> pd.DataFrame:
    df = pd.DataFrame(data["results"])
    if "rouge" in df.columns:
        df["rouge_l"] = df["rouge"].apply(lambda v: v["rougeL"] if isinstance(v, dict) else None)
    if "bertscore" in df.columns:
        df["bertscore_f1"] = df["bertscore"].apply(lambda v: v["f1"] if isinstance(v, dict) else None)
    return df


def hit_icon(v):
    return "✅" if v else "❌"


def render_single_run(data: dict, mode_label: str):
    summary = data["summary"]
    df = to_df(data)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Questions evaluated", summary["num_questions"])
    c2.metric("Retrieval hit rate", f"{summary['retrieval_hit_rate']*100:.0f}%")
    c3.metric("Mean Reciprocal Rank", summary["mean_reciprocal_rank"])
    coverage = summary.get("avg_answer_keyword_coverage")
    c4.metric("Answer keyword coverage", f"{coverage*100:.0f}%" if coverage is not None else "N/A")

    c5, c6, c7, c8 = st.columns(4)
    rouge_l = summary.get("avg_rouge_l")
    c5.metric("Avg ROUGE-L", f"{rouge_l:.2f}" if rouge_l is not None else "N/A")
    bertscore_f1 = summary.get("avg_bertscore_f1")
    c6.metric("Avg BERTScore F1", f"{bertscore_f1:.2f}" if bertscore_f1 is not None else "N/A")
    judge_score = summary.get("avg_llm_judge_score")
    c7.metric("Avg LLM Judge (1-5)", f"{judge_score:.2f}" if judge_score is not None else "N/A")
    c8.metric("Avg latency", f"{summary['avg_latency_sec']}s")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Confidence Distribution")
        conf_df = pd.DataFrame(
            list(summary["confidence_distribution"].items()), columns=["Confidence", "Count"]
        )
        st.bar_chart(conf_df.set_index("Confidence"))

    with col2:
        st.subheader("Latency per Question")
        st.bar_chart(df.set_index("id")["latency_sec"])

    st.divider()
    st.subheader("Answer-Quality Metrics by Question")
    st.caption(
        "ROUGE-L and BERTScore F1 are 0-1 lexical/semantic overlap with the reference answer. "
        "LLM Judge is a 1-5 correctness rating, shown here divided by 5 so all three sit on the same axis."
    )
    score_cols = [c for c in ["rouge_l", "bertscore_f1", "llm_judge_score"] if c in df.columns]
    if score_cols:
        chart_df = df.set_index("id")[score_cols].copy()
        if "llm_judge_score" in chart_df.columns:
            chart_df["llm_judge_score"] = chart_df["llm_judge_score"] / 5
        chart_df = chart_df.rename(columns={
            "rouge_l": "ROUGE-L",
            "bertscore_f1": "BERTScore F1",
            "llm_judge_score": "LLM Judge (/5)",
        })
        st.bar_chart(chart_df)
    else:
        st.info("No reference answers (`expected_answer`) found in the testset, so ROUGE/BERTScore/LLM-judge weren't computed.")

    st.divider()
    st.subheader("Per-Question Results")

    display_df = df.copy()
    display_df["retrieval_hit"] = display_df["retrieval_hit"].apply(hit_icon)
    display_df["keyword_coverage"] = display_df["keyword_coverage"].apply(
        lambda v: f"{v*100:.0f}%" if v is not None else "N/A"
    )
    for col in ["rouge_l", "bertscore_f1"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda v: f"{v:.2f}" if v is not None else "N/A")
    if "llm_judge_score" in display_df.columns:
        display_df["llm_judge_score"] = display_df["llm_judge_score"].apply(
            lambda v: f"{v}/5" if v is not None else "N/A"
        )

    table_cols = ["id", "question", "retrieval_hit", "keyword_coverage", "confidence", "latency_sec"]
    table_cols += [c for c in ["rouge_l", "bertscore_f1", "llm_judge_score"] if c in display_df.columns]
    st.dataframe(
        display_df[table_cols],
        use_container_width=True,
        hide_index=True,
    )

    st.divider()
    st.subheader("Inspect a Question")
    selected_id = st.selectbox("Select question", df["id"].tolist(), key=f"inspect_{mode_label}")
    row = df[df["id"] == selected_id].iloc[0]

    st.markdown(f"**Question:** {row['question']}")
    st.markdown(f"**Answer:**\n\n{row['answer']}")
    if row.get("reference_answer"):
        st.markdown(f"**Reference (gold) answer:**\n\n{row['reference_answer']}")
    st.markdown(f"**Expected source:** `{row['expected_source']}`  |  **Retrieved sources:** `{row['retrieved_sources']}`")
    st.markdown(f"**Confidence:** {row['confidence']} ({row['confidence_score']})")
    st.markdown(f"**Matched keywords:** {row['matched_keywords']} / expected {row['expected_keywords']}")

    if row.get("rouge") or row.get("bertscore") or row.get("llm_judge_score") is not None:
        st.markdown("**Answer-quality scores:**")
        sc1, sc2, sc3 = st.columns(3)
        rouge = row.get("rouge")
        sc1.metric("ROUGE-L", f"{rouge['rougeL']:.2f}" if rouge else "N/A")
        bertscore = row.get("bertscore")
        sc2.metric("BERTScore F1", f"{bertscore['f1']:.2f}" if bertscore else "N/A")
        sc3.metric("LLM Judge", f"{row['llm_judge_score']}/5" if row.get("llm_judge_score") is not None else "N/A")
        if row.get("llm_judge_reasoning"):
            st.markdown(f"*Judge reasoning: {row['llm_judge_reasoning']}*")


def _fmt_pct(v):
    return f"{v*100:.1f}%" if v is not None else "N/A"


def _fmt_num(v, decimals=3):
    return f"{v:.{decimals}f}" if v is not None else "N/A"


def render_comparison(free_data: dict, paid_data: dict):
    free_summary, paid_summary = free_data["summary"], paid_data["summary"]

    if free_summary["num_questions"] != paid_summary["num_questions"]:
        st.warning(
            f"Free run has {free_summary['num_questions']} questions, paid run has "
            f"{paid_summary['num_questions']} -- they were likely run against different versions "
            "of eval_testset.json. Comparison below still works but treat mismatched rows with care."
        )

    st.subheader("Summary: Free vs. Paid")
    rows = [
        ("Retrieval hit rate", free_summary["retrieval_hit_rate"], paid_summary["retrieval_hit_rate"], _fmt_pct, 1),
        ("Mean Reciprocal Rank", free_summary["mean_reciprocal_rank"], paid_summary["mean_reciprocal_rank"], _fmt_num, 1),
        ("Answer keyword coverage", free_summary.get("avg_answer_keyword_coverage"), paid_summary.get("avg_answer_keyword_coverage"), _fmt_pct, 1),
        ("Avg ROUGE-L", free_summary.get("avg_rouge_l"), paid_summary.get("avg_rouge_l"), _fmt_num, 1),
        ("Avg BERTScore F1", free_summary.get("avg_bertscore_f1"), paid_summary.get("avg_bertscore_f1"), _fmt_num, 1),
        ("Avg LLM Judge (1-5)", free_summary.get("avg_llm_judge_score"), paid_summary.get("avg_llm_judge_score"), _fmt_num, 1),
        ("Avg latency (s)", free_summary["avg_latency_sec"], paid_summary["avg_latency_sec"], _fmt_num, -1),
    ]
    table = []
    for label, f_val, p_val, fmt, better_dir in rows:
        delta = (p_val - f_val) if (f_val is not None and p_val is not None) else None
        if delta is None:
            delta_str = "N/A"
        else:
            sign = "+" if delta > 0 else ""
            improved = (delta > 0 and better_dir > 0) or (delta < 0 and better_dir < 0)
            arrow = " 🟢" if improved and abs(delta) > 1e-9 else (" 🔴" if abs(delta) > 1e-9 else "")
            delta_str = f"{sign}{fmt(delta) if fmt is not _fmt_pct else sign + f'{delta*100:.1f}pp'}{arrow}" if fmt is not _fmt_pct else f"{sign}{delta*100:.1f}pp{arrow}"
        table.append({"Metric": label, "Free": fmt(f_val), "Paid": fmt(p_val), "Δ (Paid - Free)": delta_str})
    st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)
    st.caption("🟢 paid is better on this metric · 🔴 paid is worse · lower latency counts as better.")

    st.divider()
    st.subheader("LLM Judge Score per Question: Free vs. Paid")
    free_df, paid_df = to_df(free_data), to_df(paid_data)
    merged = free_df[["id", "llm_judge_score"]].merge(
        paid_df[["id", "llm_judge_score"]], on="id", how="outer", suffixes=("_free", "_paid")
    ).set_index("id")
    merged = merged.rename(columns={"llm_judge_score_free": "Free", "llm_judge_score_paid": "Paid"})
    st.bar_chart(merged)

    st.divider()
    st.subheader("Per-Question Comparison")
    merged_full = free_df.merge(paid_df, on=["id", "question"], how="outer", suffixes=("_free", "_paid"))

    def diff_icon(row, col):
        f, p = row.get(f"{col}_free"), row.get(f"{col}_paid")
        if f is None or p is None:
            return ""
        if p > f:
            return "🟢 paid higher"
        if p < f:
            return "🔴 free higher"
        return "= tie"

    display_rows = []
    for _, row in merged_full.iterrows():
        display_rows.append({
            "id": row["id"],
            "question": row["question"],
            "free: hit": hit_icon(row.get("retrieval_hit_free")) if pd.notna(row.get("retrieval_hit_free")) else "N/A",
            "paid: hit": hit_icon(row.get("retrieval_hit_paid")) if pd.notna(row.get("retrieval_hit_paid")) else "N/A",
            "free: judge": row.get("llm_judge_score_free"),
            "paid: judge": row.get("llm_judge_score_paid"),
            "judge Δ": diff_icon(row, "llm_judge_score"),
            "free: rougeL": row.get("rouge_l_free"),
            "paid: rougeL": row.get("rouge_l_paid"),
            "free: bertF1": row.get("bertscore_f1_free"),
            "paid: bertF1": row.get("bertscore_f1_paid"),
            "free: latency": row.get("latency_sec_free"),
            "paid: latency": row.get("latency_sec_paid"),
        })
    st.dataframe(pd.DataFrame(display_rows), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Inspect a Question (side by side)")
    selected_id = st.selectbox("Select question", merged_full["id"].tolist(), key="inspect_compare")
    frow = free_df[free_df["id"] == selected_id]
    prow = paid_df[paid_df["id"] == selected_id]

    qcol1, qcol2 = st.columns(2)
    for col, side_df, label in [(qcol1, frow, "Free"), (qcol2, prow, "Paid")]:
        with col:
            st.markdown(f"### {label}")
            if side_df.empty:
                st.info("No result for this question in this run.")
                continue
            r = side_df.iloc[0]
            st.markdown(f"**Answer:**\n\n{r['answer']}")
            st.markdown(f"**Retrieved:** `{r['retrieved_sources']}`")
            st.markdown(f"**Confidence:** {r['confidence']} ({r['confidence_score']})")
            rouge = r.get("rouge")
            bertscore = r.get("bertscore")
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("ROUGE-L", f"{rouge['rougeL']:.2f}" if rouge else "N/A")
            sc2.metric("BERTScore F1", f"{bertscore['f1']:.2f}" if bertscore else "N/A")
            sc3.metric("Judge", f"{r['llm_judge_score']}/5" if pd.notna(r.get("llm_judge_score")) else "N/A")
            if r.get("llm_judge_reasoning"):
                st.caption(f"Judge reasoning: {r['llm_judge_reasoning']}")

    ref = frow.iloc[0].get("reference_answer") if not frow.empty else (prow.iloc[0].get("reference_answer") if not prow.empty else None)
    if ref:
        st.markdown(f"**Reference (gold) answer:**\n\n{ref}")


free_data = load_results("free")
paid_data = load_results("paid")

if not free_data and not paid_data:
    st.warning(
        "No evaluation results found yet. Run the evaluation harness first:\n\n"
        "`python -m eval.evaluate` (with RAG_MODE=free or RAG_MODE=paid in .env)"
    )
    st.stop()

tab_labels = []
if free_data:
    tab_labels.append("Free run")
if paid_data:
    tab_labels.append("Paid run")
if free_data and paid_data:
    tab_labels.append("Compare Free vs Paid")

tabs = st.tabs(tab_labels)
tab_idx = 0

if free_data:
    with tabs[tab_idx]:
        render_single_run(free_data, "free")
    tab_idx += 1

if paid_data:
    with tabs[tab_idx]:
        render_single_run(paid_data, "paid")
    tab_idx += 1

if free_data and paid_data:
    with tabs[tab_idx]:
        render_comparison(free_data, paid_data)
