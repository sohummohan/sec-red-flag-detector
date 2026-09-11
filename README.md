# SEC Filing Red-Flag Detector

A tool that pulls a public company's SEC filings straight from EDGAR and
scores each fiscal year for signs of aggressive or manipulated accounting —
combining a classic academic fraud model with NLP analysis of the filing's
own narrative language.

## Why

Most "AI + finance" student projects predict stock prices, which is both
overdone and largely (provably) not something historical price data alone
can do well. This instead asks a narrower, answerable question: **does a
company's own numbers and language contradict each other, or change in
ways that historically precede accounting scandals?** That's a real
problem investors, auditors, and regulators care about, and it can be
validated against real, documented fraud cases.

## How it works

Two independent signal sources, merged into one composite score per fiscal year:

**1. Beneish M-Score (financial).** An 8-ratio statistical model from
accounting research (Beneish, 1999) that compares year-over-year changes in
receivables, margins, asset composition, depreciation, leverage, and
accruals. It's the model that flagged Enron before its collapse. Computed
here from SEC XBRL "company facts" data — the same structured numbers
EDGAR extracts from every filed financial statement.

**2. Filing-text analysis (NLP).** For each 10-K, the tool extracts Item 1A
(Risk Factors) and Item 7 (MD&A), then computes:
- Frequency of negative / uncertain / litigious language (per 1,000 words)
- Year-over-year cosine similarity (TF-IDF) — a sharp drop means the section
  was substantially rewritten, which sometimes coincides with something
  management had to explain differently.

The two signal sets are z-scored and combined into a 0–100 composite score,
squashed with a logistic function.

## Validation

Fiscal year 2015 for **Under Armour (UAA)** — the exact year the SEC's 2021
enforcement action found the company had "pulled forward" sales from future
quarters to hit analyst estimates — scores as the single highest M-Score
(and highest composite risk score) across the company's entire 18-year
filing history. That's not tuned in; it falls out of the model on real
data. See `src/beneish.py` for the model and `app.py` for the live chart.

## Known limitations (read before treating this as more than it is)

- **The composite score is relative to each company's own filing history**,
  not calibrated across companies. Every company will show *some* year as
  its "highest risk" simply because it's the max of a finite series — that
  is not, by itself, evidence of fraud. Cross-company calibration would
  need a labeled dataset of confirmed fraud vs. non-fraud filings (the
  SEC's AAER releases are a reasonable source) and is the natural next step.
- **The text word lists are a hand-curated subset** inspired by the
  Loughran-McDonald financial sentiment dictionary, not the full academic
  list.
- **Section extraction is regex-based** and can occasionally mis-extract
  Item 1A/7 boundaries on unusually formatted filings.
- This is a research/educational heuristic, not investment advice, and
  should never be the sole basis for a real accusation or investment
  decision.

## Running it

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then edit .env with your own name + email
streamlit run app.py
```

SEC requires every EDGAR API request to identify who's making it via the
User-Agent header — that's what `.env` is for. Without it you'll get
throttled or blocked.

## Project structure

```
src/
  edgar_client.py   # SEC EDGAR API client (tickers, filings, XBRL facts)
  xbrl_utils.py      # extracts clean annual financial line items from XBRL
  beneish.py         # the Beneish M-Score model
  text_signals.py    # 10-K section extraction, tone scoring, similarity
  pipeline.py        # ties it all together into the composite score
app.py               # Streamlit dashboard
```

## Ideas for extending this

- Calibrate against the SEC's AAER dataset to turn "relative risk" into
  something closer to an actual probability.
- Swap TF-IDF similarity for sentence embeddings for more semantic
  (not just lexical) rewrite detection.
- Add a peer-comparison mode (score a company against its industry, not
  just its own history).
