"""
Answer-quality metrics: ROUGE, BERTScore, and LLM-as-judge.

All three compare a generated answer against the gold `expected_answer` in
the testset (see eval/eval_testset.json). Each is a different lens on the
same question -- "does this answer say what the reference answer says":
  - ROUGE: lexical n-gram/subsequence overlap. Cheap, deterministic, but
    penalizes correct answers that are phrased differently from the
    reference (this RAG assistant paraphrases and adds citations, so
    expect ROUGE scores well below 1.0 even on correct answers).
  - BERTScore: semantic similarity via contextual embeddings. More robust
    to paraphrasing than ROUGE, still deterministic, still no notion of
    factual correctness beyond similarity to the reference text.
  - LLM-as-judge: an LLM call that reads the question, reference answer,
    and candidate answer, and scores correctness/groundedness directly.
    Non-deterministic and costs an API call per question, but is the only
    one of the three that can judge whether the candidate is actually
    *right*, not just textually similar to the reference.
"""
from __future__ import annotations

import json
import re
from typing import Any

import config
from rag.llm_client import chat_complete

_rouge_scorer = None
_bertscore_model_cache: dict[str, Any] = {}


def _get_rouge_scorer():
    global _rouge_scorer
    if _rouge_scorer is None:
        from rouge_score import rouge_scorer
        _rouge_scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )
    return _rouge_scorer


def compute_rouge(reference: str, candidate: str) -> dict[str, float]:
    """Returns F-measure for rouge1/rouge2/rougeL, each in [0, 1]."""
    scorer = _get_rouge_scorer()
    scores = scorer.score(reference, candidate)
    return {
        "rouge1": round(scores["rouge1"].fmeasure, 4),
        "rouge2": round(scores["rouge2"].fmeasure, 4),
        "rougeL": round(scores["rougeL"].fmeasure, 4),
    }


def compute_bertscore_batch(candidates: list[str], references: list[str]) -> list[dict[str, float]]:
    """Batched BERTScore -- loads the model once for the whole eval run rather
    than per-question, since model load dominates cost on CPU."""
    import bert_score

    P, R, F1 = bert_score.score(
        candidates,
        references,
        lang=config.EVAL_BERTSCORE_LANG,
        model_type=config.EVAL_BERTSCORE_MODEL,
        verbose=False,
    )
    return [
        {
            "precision": round(p.item(), 4),
            "recall": round(r.item(), 4),
            "f1": round(f.item(), 4),
        }
        for p, r, f in zip(P, R, F1)
    ]


_JUDGE_SYSTEM_PROMPT = """You are an impartial evaluator scoring a RAG assistant's answer \
against a reference (gold) answer. Score the CANDIDATE answer on a 1-5 scale for how well \
it conveys the same facts as the REFERENCE answer, given the QUESTION:

5 = Fully correct: all key facts in the reference are present and correct, no contradictions.
4 = Mostly correct: minor omission or imprecision, but no wrong facts.
3 = Partially correct: some key facts present, but missing important ones or partially wrong.
2 = Mostly incorrect: only a small part of the reference is reflected, or notable errors.
1 = Incorrect or non-answer: contradicts the reference, hallucinates facts, or fails to answer.

Respond with ONLY a JSON object, no other text:
{"score": <integer 1-5>, "reasoning": "<one or two sentence justification>"}
"""


def llm_judge(question: str, reference_answer: str, candidate_answer: str) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"QUESTION:\n{question}\n\n"
                f"REFERENCE ANSWER:\n{reference_answer}\n\n"
                f"CANDIDATE ANSWER:\n{candidate_answer}"
            ),
        },
    ]
    # Generous max_tokens: reasoning-style models (e.g. the free-tier
    # openai/gpt-oss-120b via OpenRouter) spend part of the budget on hidden
    # reasoning before emitting the JSON, and return empty content if cut off.
    raw = chat_complete(messages, temperature=config.EVAL_JUDGE_TEMPERATURE, max_tokens=600)

    if not raw:
        return {"score": None, "reasoning": "[empty judge response -- likely truncated before completion]"}

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return {"score": None, "reasoning": f"[unparseable judge response] {raw[:200]}"}
    try:
        parsed = json.loads(match.group(0))
        score = int(parsed.get("score"))
        if not 1 <= score <= 5:
            raise ValueError("score out of range")
        return {"score": score, "reasoning": str(parsed.get("reasoning", ""))}
    except (json.JSONDecodeError, TypeError, ValueError):
        return {"score": None, "reasoning": f"[unparseable judge response] {raw[:200]}"}
