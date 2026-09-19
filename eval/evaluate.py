"""
Evaluation harness.

Computes, per test question:
  - Retrieval hit@k: was the expected source document among the retrieved chunks?
  - MRR (mean reciprocal rank) of the expected source in the ranked results.
  - Answer keyword coverage: does the generated answer contain the expected
    key facts (a cheap proxy that doesn't need a reference answer at all).
  - ROUGE-1/2/L, BERTScore, and an LLM-as-judge score, each comparing the
    generated answer against the gold `expected_answer` in the testset (see
    eval/metrics.py for what each one actually measures).
  - Confidence label returned, and end-to-end latency.

Aggregates into summary metrics and writes eval_results.json, which the
Streamlit eval dashboard (eval/dashboard.py) visualizes.

Run:
    python -m eval.evaluate
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import config
from rag.vector_store import VectorStore
from rag.retriever import retrieve_and_rerank
from rag.generator import generate_answer
from eval.metrics import compute_rouge, compute_bertscore_batch, llm_judge


def load_testset(path: str = config.EVAL_TESTSET_PATH) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def evaluate_question(item: dict, store: VectorStore) -> dict:
    t0 = time.time()
    chunks = retrieve_and_rerank(item["question"], store)
    result = generate_answer(item["question"], chunks)
    latency = time.time() - t0

    retrieved_sources = [c["metadata"].get("source") for c in chunks]
    expected = item.get("expected_source")

    if expected is None:
        # negative test case: correct behavior is NOT finding the doc / hedging
        hit = expected not in retrieved_sources or True  # source absence isn't itself a failure
        rank = None
        retrieval_correct = True  # judged via answer coverage instead
    else:
        hit = expected in retrieved_sources
        rank = retrieved_sources.index(expected) + 1 if hit else None
        retrieval_correct = hit

    mrr = (1 / rank) if rank else 0.0

    answer_lower = result["answer"].lower()
    keywords = item.get("expected_answer_contains", [])
    matched_keywords = [kw for kw in keywords if kw.lower() in answer_lower]
    keyword_coverage = len(matched_keywords) / len(keywords) if keywords else None

    reference_answer = item.get("expected_answer")
    rouge = compute_rouge(reference_answer, result["answer"]) if reference_answer else None
    judge = llm_judge(item["question"], reference_answer, result["answer"]) if reference_answer else None

    return {
        "id": item["id"],
        "question": item["question"],
        "expected_source": expected,
        "retrieved_sources": retrieved_sources,
        "retrieval_hit": retrieval_correct,
        "mrr": mrr,
        "answer": result["answer"],
        "reference_answer": reference_answer,
        "confidence": result["confidence"],
        "confidence_score": result["confidence_score"],
        "keyword_coverage": keyword_coverage,
        "matched_keywords": matched_keywords,
        "expected_keywords": keywords,
        "rouge": rouge,
        "llm_judge_score": judge["score"] if judge else None,
        "llm_judge_reasoning": judge["reasoning"] if judge else None,
        "latency_sec": round(latency, 2),
    }


def run_evaluation(testset_path: str = config.EVAL_TESTSET_PATH, out_path: str = config.EVAL_RESULTS_PATH) -> dict:
    store = VectorStore()
    if store.count() == 0:
        raise RuntimeError("Vector store is empty. Run ingestion first: python -m ingestion.ingest")

    testset = load_testset(testset_path)
    results = [evaluate_question(item, store) for item in testset]

    # BERTScore loads a model, so compute it once for the whole batch rather
    # than per-question.
    scoreable = [(i, r) for i, r in enumerate(results) if r["reference_answer"]]
    if scoreable:
        candidates = [r["answer"] for _, r in scoreable]
        references = [r["reference_answer"] for _, r in scoreable]
        bertscores = compute_bertscore_batch(candidates, references)
        for (i, _), bs in zip(scoreable, bertscores):
            results[i]["bertscore"] = bs
    for r in results:
        r.setdefault("bertscore", None)

    n = len(results)
    hit_rate = sum(r["retrieval_hit"] for r in results) / n
    avg_mrr = sum(r["mrr"] for r in results) / n
    coverage_scores = [r["keyword_coverage"] for r in results if r["keyword_coverage"] is not None]
    avg_keyword_coverage = sum(coverage_scores) / len(coverage_scores) if coverage_scores else None
    avg_latency = sum(r["latency_sec"] for r in results) / n
    confidence_dist = {}
    for r in results:
        confidence_dist[r["confidence"]] = confidence_dist.get(r["confidence"], 0) + 1

    rouge_l_scores = [r["rouge"]["rougeL"] for r in results if r["rouge"]]
    avg_rouge_l = sum(rouge_l_scores) / len(rouge_l_scores) if rouge_l_scores else None
    bertscore_f1_scores = [r["bertscore"]["f1"] for r in results if r["bertscore"]]
    avg_bertscore_f1 = sum(bertscore_f1_scores) / len(bertscore_f1_scores) if bertscore_f1_scores else None
    judge_scores = [r["llm_judge_score"] for r in results if r["llm_judge_score"] is not None]
    avg_llm_judge_score = sum(judge_scores) / len(judge_scores) if judge_scores else None

    summary = {
        "num_questions": n,
        "retrieval_hit_rate": round(hit_rate, 3),
        "mean_reciprocal_rank": round(avg_mrr, 3),
        "avg_answer_keyword_coverage": round(avg_keyword_coverage, 3) if avg_keyword_coverage is not None else None,
        "avg_rouge_l": round(avg_rouge_l, 3) if avg_rouge_l is not None else None,
        "avg_bertscore_f1": round(avg_bertscore_f1, 3) if avg_bertscore_f1 is not None else None,
        "avg_llm_judge_score": round(avg_llm_judge_score, 2) if avg_llm_judge_score is not None else None,
        "avg_latency_sec": round(avg_latency, 2),
        "confidence_distribution": confidence_dist,
    }

    output = {"summary": summary, "results": results}
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\n[DONE] Full results written to {out_path}")
    return output


if __name__ == "__main__":
    run_evaluation()
