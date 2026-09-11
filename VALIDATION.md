# Validation Study

A single anecdote (Under Armour FY2015) is a demo, not evidence. This is
an attempt at something closer to evidence: 4 documented SEC enforcement
cases, verified against primary sources, plus 8 industry-matched "clean"
companies as controls, run through the same pipeline every other company
in the app runs through.

Reproduce with:
```bash
python src/validation_study.py --max-filings 10 --out validation_results.csv
```

## The dataset

Every fraud case's flagged fiscal years were checked directly against the
SEC's own press release (not secondary news coverage) via the citation
links below — two of the four required correcting an initial estimate
that came from paraphrased secondary sources (see "Corrections" below).

| Ticker | Company | Flagged fiscal years | SEC source |
|---|---|---|---|
| UAA | Under Armour | 2015, 2016 | [2021-78](https://www.sec.gov/newsroom/press-releases/2021-78) |
| KHC | Kraft Heinz | 2016, 2017, 2018 | [2021-174](https://www.sec.gov/newsroom/press-releases/2021-174) |
| MDXG | MiMedx | 2013-2017 | [2019-243](https://www.sec.gov/newsroom/press-releases/2019-243) |
| GVA | Granite Construction | 2017, 2018, 2019 | [2022-150](https://www.sec.gov/newsroom/press-releases/2022-150) |

Note on wording: Under Armour's is a **disclosure failure** finding, not
an alleged GAAP violation — the SEC was explicit that it didn't allege the
underlying sales violated accounting rules. It's included because pulling
sales forward is exactly the kind of revenue-timing behavior the Beneish
ratios (DSRI, SGI) are built to catch, regardless of which side of the
GAAP line the SEC placed it on.

Clean controls (same industry, no known SEC accounting enforcement action):
Nike & Columbia Sportswear (vs. UAA), General Mills & Campbell's (vs. KHC),
Integra LifeSciences & Organogenesis (vs. MDXG), Fluor & AECOM (vs. GVA).

## Methodology

Two scores, evaluated two ways:

1. **Composite risk score (0-100)** — z-scored Beneish ratios + text tone/
   rewrite signals, logistic-squashed, normalized against each company's
   *own* history (see README for why: there's no cross-company calibration
   data here).
2. **Hotelling T² / Ledoit-Wolf** — a multivariate control-chart statistic
   over the 8 Beneish ratios (see "Industrial engineering framing" in the
   README). `t2/UCL` is comparable across companies, unlike the composite
   score.

Because the composite score is normalized per-company, a fixed 0-100
threshold isn't directly comparable across companies — every company has
*some* highest year by construction. So we report two different, both
legitimate, ways of reading the results:

- **Within-company rank**: did the documented fraud year land as the
  single riskiest year in that company's own multi-year history?
- **Pooled ROC-AUC / confusion matrix**: treating every company-year
  (fraud and clean) as one pool and asking whether the score, on its own
  numeric scale, separates the two groups.

## Results

| Metric | Pooled ROC-AUC | Sensitivity @ threshold | Specificity @ threshold |
|---|---|---|---|
| Composite score (threshold 70) | 0.424 | 16.7% (2/12 flagged years) | 94.7% |
| Hotelling T²/UCL (threshold 1.0) | 0.253 | 0.0% (0/8) | 96.5% |

Pooled AUC for the composite score is close to random, **because it isn't
designed to be read on an absolute cross-company scale** — see the
within-company view below, which is the metric the score was actually
built for:

| Company | Flagged-year max composite score | Rank within own history |
|---|---|---|
| UAA | 86.4 | **1 / 16** |
| KHC | 66.6 | **1 / 10** |
| GVA | 77.0 | **1 / 11** |
| MDXG | 49.2 | 7 / 14 |

**3 of 4 documented cases land their fraud year as the single highest-risk
year the model has ever assigned that company**, against roughly 0.33
expected hits if the model had no signal at all (1/16 + 1/10 + 1/11 + 1/14
≈ 0.33, summing the chance of landing on the single top rank by luck in
each company's own history). That's a real signal, on a genuinely tiny
sample — four cases is not enough to make a statistical claim, but it's
enough to say the model is doing *something* right and not merely curve-
fitted to Under Armour.

The Hotelling T² metric, despite being the more sophisticated statistical
technique, did not outperform the simpler score here (see "What didn't
work" below) — worth stating plainly rather than presenting only the
technique that happened to look good.

## The MiMedx miss

MiMedx's raw Beneish M-Score does cross the classic fraud threshold
(-1.78) in 2 of its 5 flagged years (FY2013: -1.56, FY2015: -1.22) — the
financial signal is there. But the *composite* score, which folds in
text-tone and rewrite analysis, buries it: MiMedx's fraud was undisclosed
side arrangements with distributors affecting revenue recognition, not
necessarily the kind of conduct that shows up as unusual Risk Factors
language. Folding financial and textual signals together with equal
weight likely hurts more than it helps for revenue-recognition-style
fraud, where the risk factors section can stay perfectly ordinary while
the accounting quietly diverges from it. A weighted or type-aware
combination (up-weighting the financial signal when the two disagree,
rather than always averaging) is a natural next iteration.

## What didn't work (and why that's worth keeping)

Getting to the numbers above took three real bugs, caught only because
the first version's *results* looked internally inconsistent enough to
go digging:

1. **Quarterly data leaking into "annual" figures.** Kraft Heinz's 2019
   10-K (filed after the SEC investigation began) included a footnote
   reconciling every quarter of 2017-2018 — each tagged in SEC's XBRL data
   with the same `form: "10-K"` metadata as the real annual balance sheet
   figures. The initial extraction couldn't tell them apart, so KHC's
   "annual" financial history briefly included spurious quarter-end rows.
   Fixed by cross-checking every extracted fact's date against the
   company's actual list of 10-K report dates (`src/xbrl_utils.py`).
2. **Division-by-zero producing `inf`, not `NaN`.** A spuriously-tagged
   $0 receivables value crashed the covariance estimator outright for
   MiMedx. `inf` from a zero denominator means "this ratio is undefined,"
   and should be treated exactly like missing data (`src/beneish.py`).
3. **Naive sample covariance for Hotelling T².** With only ~10-15 years of
   history to estimate an 8-dimensional covariance matrix, the raw sample
   covariance was wildly unstable — General Mills, a company with no known
   accounting issues, briefly showed a T²/UCL ratio over 9,000. Ledoit-Wolf
   shrinkage fixed the numerical blowups, but even after that fix, T² still
   underperformed the simpler composite score on this dataset (pooled AUC
   0.253, worse than random) and missed Under Armour's flagged year
   entirely (T²/UCL 0.46, under its own control limit). The honest
   conclusion: this technique, textbook-correct as it is, needs either more
   historical years per company or fewer than 8 dimensions (a PCA-reduced
   version is a reasonable next thing to try) before it adds value here.
4. **Fraud-year labels initially came from paraphrased secondary
   coverage**, not the SEC's own words. Re-checking the actual press
   releases changed Kraft Heinz's window from [2016, 2017] to [2016,
   2017, 2018] and MiMedx's from [2013-2016] to [2013-2017] — and the KHC
   correction flipped that case from an apparent miss (rank 9/10) to a
   hit (rank 1/10), because the excluded FY2018 turned out to be the
   company's actual highest-scoring year. A wrong label would have quietly
   produced a wrong conclusion in either direction.

None of this is a weakness to hide — a validation study whose first draft
was quietly wrong in three different ways, caught before being reported
as a finished result, is a stronger claim than one that happened to work
on the first try.

## Honest limitations

- n=4 fraud cases is too small to support a real precision/recall claim.
  Expanding this dataset (more SEC AAER cases, more controls) is the
  highest-leverage next step for making this claim more rigorous.
- All 4 fraud companies are large/mid-cap US-listed firms with full XBRL
  history; the model hasn't been tested on smaller filers with thinner
  XBRL tag coverage (several of our own clean-company runs already show
  gaps in less-common tags like `PropertyPlantAndEquipmentNet` or
  `DepreciationDepletionAndAmortization`).
- The composite score's per-company normalization means it cannot say
  "Company A is riskier than Company B" — only "this is an unusual year
  *for this company*." The T²/UCL ratio was an attempt at a cross-company
  comparable metric; it isn't there yet (see above).
