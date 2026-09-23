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

**3. Multivariate statistical process control (Hotelling T²).** The
composite score above treats the 8 Beneish ratios independently; T²
instead treats them as one multivariate "process" and measures how far a
given year sits from the company's own historical center *accounting for
how the ratios normally move together*. See "Industrial engineering
framing" below.

## Validation

Fiscal year 2015 for **Under Armour (UAA)** — the SEC's 2021 settlement found
the company failed to disclose that it had pulled forward $408M in sales
across six quarters starting Q3 2015 (a disclosure failure; the SEC did not
allege the sales themselves violated GAAP) — scores as the single highest
M-Score across the company's entire 18-year filing history. That single
result isn't tuned in, but it's one anecdote.

**[VALIDATION.md](VALIDATION.md)** has the real study: 4 documented SEC
enforcement cases (Under Armour, Kraft Heinz, MiMedx, Granite Construction —
every flagged fiscal year checked against the SEC's own press releases, not
secondhand summaries) plus 8 industry-matched clean companies, run through
`src/validation_study.py`. Headline result: **3 of 4 fraud cases land their
documented fraud year as the single riskiest year in that company's own
filing history** — against roughly 0.33 expected by chance. It also
documents where the model missed (MiMedx) and three real bugs found and
fixed along the way, including one that had initially made a result look
better than it actually was.

## Industrial engineering framing

The composite score above is a fairly ad hoc heuristic (average some
z-scores, squash with a logistic). `src/spc.py` implements a more
principled alternative: **Hotelling's T²**, the standard technique from
industrial engineering quality control (e.g. UC Berkeley's
[IEOR 165](https://ieor.berkeley.edu), *Engineering Statistics, Quality
Control, and Forecasting*) for monitoring a multivariate process — normally
a manufacturing line, here a company's own reported financial ratios.
Because 8 ratios estimated from only ~10-15 years of company history is a
classic high-dimension/low-sample-size covariance estimation problem, the
raw sample covariance was unusable (it produced a T²/UCL ratio over 9,000
for a company with no known accounting issues); `src/spc.py` uses
Ledoit-Wolf shrinkage instead, the standard covariance-regularization
technique from financial risk modeling (the kind of thing covered in
Berkeley's [IEOR 241](https://ieor.berkeley.edu), *Risk Modeling,
Simulation, and Data Analysis*) for exactly this problem in portfolio
covariance estimation.

Worth saying plainly: on this dataset, T² did **not** outperform the
simpler composite score (see VALIDATION.md) — it's included because it's
a legitimate, more statistically rigorous method worth knowing how to
apply, not because it won. A negative result from the more sophisticated
technique is still a real result.

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

### Deploying on Streamlit Community Cloud

Cloud deployments have no `.env` file, so set the same value as a secret in
the app's **Settings → Secrets** panel:

```toml
SEC_EDGAR_USER_AGENT = "Your Name your.email@example.com"
```

`app.py` reads it from `st.secrets` automatically.

## Project structure

```
src/
  edgar_client.py     # SEC EDGAR API client (tickers, filings, XBRL facts)
  xbrl_utils.py       # extracts clean annual financial line items from XBRL
  beneish.py          # the Beneish M-Score model
  text_signals.py     # 10-K section extraction, tone scoring, similarity
  spc.py              # Hotelling T-squared / Ledoit-Wolf multivariate control chart
  pipeline.py         # ties it all together into the composite score
  validation_cases.py # documented SEC fraud cases + matched clean companies
  validation_study.py # runs the pipeline across validation_cases.py, scores it
app.py                # Streamlit dashboard
VALIDATION.md         # the actual validation study, methodology + results
```

## Ideas for extending this

- Calibrate against the SEC's AAER dataset to turn "relative risk" into
  something closer to an actual probability.
- Swap TF-IDF similarity for sentence embeddings for more semantic
  (not just lexical) rewrite detection.
- Add a peer-comparison mode (score a company against its industry, not
  just its own history).
