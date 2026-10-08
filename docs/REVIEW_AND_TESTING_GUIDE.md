# Review & Testing Guide — `feature/react-agent-and-eval-report`

This guide explains how to set up, test, and review the ReAct agent / evaluation
branch before it is merged into `main`.

**Repo:** `github.com/kpareek/Enterprise_knowledge_assistant`
**Branch:** `feature/react-agent-and-eval-report`
**Target:** `main` (via a draft Pull Request)

---

## 1. What changed in this branch

| Area | Change | Files |
|---|---|---|
| ReAct agent | Thought → Action → Observation loop with multi-step retrieval and query reformulation; drop-in alternative to single-shot RAG | `rag/react_agent.py` |
| Escalation UX | Single-shot answers first; if the answer is weak, the app offers to switch to ReAct, shows the extra LLM-call cost, and runs it only if the user accepts | `app.py` |
| Calibrated confidence | Confidence now reflects groundedness — a refusal caps to 🔴 Low, a hedge to 🟡 Medium — instead of trusting the raw retrieval score | `rag/generator.py` |
| LLM client robustness | Retries transient empty free-tier responses; fails fast on hard daily-quota walls; strips leaked reasoning from answers | `rag/llm_client.py` |
| Bug fixes | Reasoning leak, ReAct scaffold leak, ReAct retrieval drift (see `docs/reasoning_leak_analysis.md`) | `rag/llm_client.py`, `rag/react_agent.py` |
| Evaluation | ReAct strategy in the eval runner, multi-hop test questions, single-vs-ReAct dashboard tab, HTML report renderer | `eval/*` |
| Reports | Free-vs-paid and ReAct-vs-single-shot comparison from measured data | `EVALUATION_REPORT.md/.html`, `SUMMARY.md/.html` |
| Setup docs | Free + paid setup so each teammate runs with their own key | `.env.example`, `README.md` |

---

## 2. API keys — use YOUR OWN key ⚠️

**Every reviewer must use their own API key. Never share keys, and never commit them.**

How the project keeps keys out of git:

- Keys are read **only** from environment variables, loaded from a local `.env`
  file by `config.py`. No key is hardcoded anywhere in the code.
- `.env` is listed in `.gitignore`, so git will not track it.
- Only `.env.example` is committed, and it contains **placeholders only**
  (`sk-or-v1-your-key-here`, `sk-your-key-here`).

### Get a key (pick one)

| Mode | Set in `.env` | Where to get the key | Cost |
|---|---|---|---|
| **Free** (recommended for review) | `RAG_MODE=free` + `OPENROUTER_API_KEY` | https://openrouter.ai/keys — no card needed | Free; **~50 requests/day** cap |
| **Paid** | `RAG_MODE=paid` + `OPENAI_API_KEY` | https://platform.openai.com/api-keys | Pay-per-use (needs credit) |

### Before you commit or push anything, verify no key is staged

```bash
git check-ignore .env          # must print ".env" — means it is ignored
git status                     # .env must NOT appear in the list
git diff --cached | grep -E 'sk-(or-v1|proj|ant)-[A-Za-z0-9_-]{20,}' && echo "STOP: key staged!"
```

**Optional safety net** — a local pre-commit hook that blocks commits containing a key:

```bash
cat > .git/hooks/pre-commit <<'EOF'
#!/bin/sh
if git diff --cached -U0 | grep -qE 'sk-(or-v1|proj|ant)-[A-Za-z0-9_-]{20,}'; then
  echo "pre-commit: an API key appears in your staged changes. Commit blocked."
  exit 1
fi
EOF
chmod +x .git/hooks/pre-commit
```

If you ever accidentally commit or paste a key somewhere shared, **revoke it
immediately** in the provider's dashboard and create a new one — deleting the
commit is not enough.

---

## 3. Setup (every reviewer, ~10 minutes)

Requires Python 3.10+.

```bash
# 1. Clone and switch to the branch
git clone git@github.com:kpareek/Enterprise_knowledge_assistant.git
cd Enterprise_knowledge_assistant
git checkout feature/react-agent-and-eval-report

# 2. Virtual environment + dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Add YOUR OWN key
cp .env.example .env
#    edit .env: set RAG_MODE and the matching key (see section 2)

# 4. Build the vector index (first run downloads a ~90MB local embedding model)
python -m ingestion.ingest --dir data/sample_docs --reset
python -m ingestion.ingest --dir data/external_enterpriserag_bench

# 5. Run the chat app → http://localhost:8501
streamlit run app.py
```

> If any step fails or the instructions are unclear, log it as review feedback —
> that is a README bug.

---

## 4. Testing

Testing is done in three layers. Layer 1 is for everyone; Layers 2 and 3 can be
split between reviewers.

### Layer 1 — Setup smoke test (everyone)

- [ ] Section 3 completes without errors on your machine
- [ ] The app opens at http://localhost:8501
- [ ] A simple question returns an answer with a confidence badge and a 📎 Sources expander

### Layer 2 — Functional checklist (1–2 reviewers)

Ask each question in the chat app and compare with the expected behavior.

| # | Question | Expected behavior | Pass? |
|---|---|---|---|
| 1 | How many days of annual leave do I accrue per year? | 21 days (1.75/month), 🟢 High, sources shown | |
| 2 | What should I do if my VPN connection keeps timing out? | Restart GlobalProtect, escalate to Helpdesk; caption shows `⚡ Single-shot · 1 LLM call` | |
| 3 | Do I need to change my password every 90 days? | **No** — the 90-day rule was removed in policy v2.0 (current version wins, not a blend) | |
| 4 | What approval is required for a $30,000 purchase? | Finance approval + 2 quotes ($10k–$50k tier) | |
| 5 | What is South Africa's standard work week? | 45 hours (from one row of an appendix table) | |
| 6 | If I quit my job, do I get paid for the vacation days I didn't use? | Yes — encashment up to the 10-day cap. **No "We need to answer…" reasoning text in the answer** (reasoning-leak fix) | |
| 7 | What is the company's sabbatical leave policy? | Honest refusal ("not in the documents"), 🔴 Low | |
| 8 | What is the employee dress code policy? | Honest refusal, 🔴 Low | |
| 9 | What is the maximum dollar allowance for a business class flight upgrade? | Says no dollar cap is specified — does **not** invent a number | |
| 10 | As a new hire, when does my health insurance start, how many annual leave days do I accrue, and what is the minimum password length? | Single-shot is weak (🔴 Low) → **escalation prompt** appears with cost → click **"🔷 Yes, use ReAct agent"** → ReAct runs multiple searches and answers all three parts, 🟢 High; caption shows `🔷 ReAct agent · N LLM calls · N searches` | |
| 11 | "How much annual leave do I get?" → then "Can I carry it forward?" → then "What about sick leave?" | Follow-ups use the conversation context | |
| 12 | Decline the escalation prompt on #10 | App keeps the single-shot answer, no extra LLM calls are made | |

### Layer 3 — Evaluation regression (1 reviewer per mode)

```bash
python -m eval.evaluate                   # score the single-shot pipeline
python -m eval.evaluate --strategy react  # score the ReAct agent
streamlit run eval/dashboard.py           # visualize → http://localhost:8502
python eval/build_report_html.py          # re-render the HTML reports
```

- [ ] Both runs complete
- [ ] Hit-rate and keyword coverage are close to the committed baselines in
      `eval/eval_results_free.json` / `eval/eval_results_paid.json`
      (free models are non-deterministic, so small differences are expected)
- [ ] ReAct scores at least as well as single-shot on the multi-hop questions
- [ ] The dashboard's single-vs-ReAct tab renders

> ⚠️ **Free-tier quota:** OpenRouter free keys allow ~50 requests/day. A full
> ReAct eval over 28 questions makes several LLM calls per question and will
> likely exceed it. Split the work — one reviewer runs single-shot, another runs
> ReAct — or run the ReAct eval in paid mode.

---

## 5. Code review — split by area

| Reviewer | Files | Focus |
|---|---|---|
| **A — Agent logic** | `rag/react_agent.py` | Loop always terminates (step cap); citation registry stays stable across steps; malformed-output fallback returns clean answer text; full-corpus search with organization filter only |
| **B — LLM & confidence** | `rag/llm_client.py`, `rag/generator.py` | Retry vs. fail-fast classification of errors; `<think>` / reasoning stripping; refusal and hedge detection in confidence scoring |
| **C — App & UX** | `app.py` | Escalation prompt wording and cost display; chat history stays consistent after escalating; per-answer strategy/cost/latency caption |
| **D — Eval & docs** | `eval/*`, `EVALUATION_REPORT.md`, `SUMMARY.md`, `docs/` | Report numbers match `eval_results_*.json`; test questions and expected keywords are fair; docs are accurate |

Review checklist for everyone:

- [ ] No API keys, tokens, or personal/company-confidential data in the diff
- [ ] Error handling is reasonable (no silent failures, no crashes on a 429)
- [ ] Code is readable and consistent with the rest of the project
- [ ] Leave comments on the PR lines; mark blocking issues clearly

---

## 6. Known issues (please don't re-report)

| Issue | Impact | Status |
|---|---|---|
| Confidence badge shows 🟢 High when the answer says "I **cannot** find…" (e.g. *"Can I bring my pet to the office?"*) — refusal detector only matches "could not find" / "couldn't find" | Wrong confidence label on some refusals | Fix planned before merge |
| No automated unit tests — verification is manual + eval runs | Regressions rely on manual checks | Planned: small `pytest` suite for confidence scoring, reasoning cleanup, ReAct fallback (no API key needed) |
| Free-tier `429 Rate limit exceeded: free-models-per-day` | Testing stops until the daily reset | Expected — wait for reset, or use paid mode |

---

## 7. Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `401 Missing Authentication header` | Key missing/empty in `.env`, or `RAG_MODE` doesn't match the key you set | Check `.env`: `RAG_MODE=free` needs `OPENROUTER_API_KEY`; `RAG_MODE=paid` needs `OPENAI_API_KEY` |
| `429 Rate limit exceeded: free-models-per-day` | OpenRouter free daily cap reached | Wait for the daily reset or switch to paid mode |
| Answers are empty / "no relevant documents" | Index not built, or built in the other mode | Re-run the ingestion commands in section 3 (paid mode uses a separate index) |
| Dashboard not reachable on 8502 | Port in use or app not started | `streamlit run eval/dashboard.py --server.port 8503` |

---

## 8. Merge criteria

The PR is merged into `main` when:

- [ ] At least one approval per review area (A–D)
- [ ] Layer 1 and Layer 2 checklists pass on at least one teammate's machine
- [ ] Layer 3 eval results are in line with the baselines
- [ ] Known issue #1 (confidence badge) is fixed
- [ ] All review comments are resolved
- [ ] No secrets in the diff (`git diff main...HEAD | grep -E 'sk-(or-v1|proj|ant)-'` returns nothing)
