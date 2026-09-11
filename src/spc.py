"""Multivariate statistical process control (Hotelling's T-squared), applied
to a company's own financial-ratio history instead of a manufacturing line.

This is the standard technique from industrial engineering quality control
(e.g. UC Berkeley IEOR 165, "Engineering Statistics, Quality Control, and
Forecasting") for detecting when a multivariate process has drifted out of
its normal operating region -- accounting for *correlation* between
variables, which our earlier equal-weighted z-score sum ignored entirely.
Two ratios moving together in their usual way shouldn't count as "twice as
anomalous" just because they're both elevated.

We treat each fiscal year's vector of 8 Beneish ratios as one multivariate
"process observation" and ask: how far is this year from the company's own
historical center, in units that account for how the ratios normally
co-vary? That's exactly Hotelling's T².

Because we don't have an independent baseline period (we're mining a
company's whole history for the anomalous year(s), a "Phase I" SPC
problem), each year's mean/covariance reference is estimated leave-one-out
from every *other* year -- so a company can't inflate its own T² by being
part of the baseline it's compared against.

A first version of this used the raw sample covariance matrix and it was
a disaster: with only ~10-15 fiscal years available to estimate an 8x8
covariance matrix, the estimate is extremely noisy -- small eigenvalues in
the sample covariance get wildly overweighted after inversion, which both
produced nonsense (a T²/UCL ratio over 9000 for General Mills, a company
with no known accounting issues) and completely missed the one case the
simpler composite score caught cleanly (Under Armour's flagged year scored
*under* its own control limit). This is the classic high-dimension/
low-sample-size covariance estimation problem. The standard fix -- also
genuinely Berkeley IEOR/financial-engineering material, since it's exactly
the technique portfolio risk models use to stabilize covariance estimates
from short return histories -- is Ledoit-Wolf shrinkage: blend the noisy
sample covariance toward a well-conditioned target (scaled identity) by an
amount chosen to minimize expected estimation error. See Ledoit & Wolf,
"Improved Estimation of the Covariance Matrix of Stock Returns," J.
Empirical Finance (2003).

Caveat: shrinking the covariance estimate means the classical F-distribution
UCL (derived for the unshrunk sample covariance) is only an approximation
here, not an exact critical value. We still report it as a rough visual
reference line, but treat the T² *magnitude* (for ranking/AUC) as the
trustworthy part of this signal, not the exact UCL crossing.
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.covariance import LedoitWolf

RATIO_COLUMNS = ["DSRI", "GMI", "AQI", "SGI", "DEPI", "SGAI", "LVGI", "TATA"]
MIN_BASELINE_YEARS = 6


def hotelling_t2(ratios: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """ratios: one row per fiscal year, columns = RATIO_COLUMNS (NaNs allowed;
    those years are skipped). Returns a DataFrame with t2, ucl, and
    out_of_control for every year that had enough history to test.
    """
    data = ratios[RATIO_COLUMNS].dropna()
    n, p = data.shape
    out = pd.DataFrame(index=ratios.index, columns=["t2", "ucl", "out_of_control"])

    if n < MIN_BASELINE_YEARS + 1:
        # not enough historical years to estimate a covariance structure
        # at all reliably -- refuse to produce a number instead of
        # returning a misleadingly precise one
        return out

    for year in data.index:
        baseline = data.drop(index=year)
        m = len(baseline)
        if m < MIN_BASELINE_YEARS:
            continue

        mean = baseline.mean().to_numpy()
        lw = LedoitWolf().fit(baseline.to_numpy())
        cov_inv = np.linalg.pinv(lw.covariance_)

        x = data.loc[year].to_numpy()
        diff = x - mean
        t2 = float(diff @ cov_inv @ diff.T)

        # approximate reference line only (see module docstring caveat on
        # shrinkage invalidating the classical exact F critical value)
        f_crit = stats.f.ppf(1 - alpha, p, max(m - p, 1))
        ucl = p * (m - 1) / max(m - p, 1) * f_crit

        out.loc[year, "t2"] = t2
        out.loc[year, "ucl"] = ucl
        out.loc[year, "out_of_control"] = t2 > ucl

    # t2/ucl is comparable *across* companies (unlike the raw composite
    # score, which is normalized against each company's own mean) --
    # >1.0 means "outside this company's own statistically-derived control
    # limit," regardless of which company it is
    out["t2_ratio"] = pd.to_numeric(out["t2"]) / pd.to_numeric(out["ucl"])
    return out
