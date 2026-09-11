"""End-to-end: ticker -> financial (Beneish) + textual red-flag signals,
merged into one composite risk score per fiscal year.
"""
import numpy as np
import pandas as pd

import beneish
import edgar_client
import spc
import text_signals
import xbrl_utils


def financial_signals(cik: str) -> pd.DataFrame:
    facts = edgar_client.get_company_facts(cik)
    valid_ends = [f["reportDate"] for f in edgar_client.list_filings(cik, "10-K")]
    df = xbrl_utils.build_annual_dataframe(facts, valid_fiscal_year_ends=valid_ends)
    return beneish.compute_m_score(df)


def textual_signals(cik: str, max_filings: int = 8) -> pd.DataFrame:
    """Fetch up to `max_filings` most recent 10-Ks, extract Item 1A/7,
    and compute tone scores + year-over-year similarity."""
    filings = edgar_client.list_filings(cik, "10-K")[:max_filings]
    filings = sorted(filings, key=lambda f: f["reportDate"])

    rows = {}
    prev_risk_text, prev_mda_text = None, None
    for f in filings:
        try:
            full_text = edgar_client.fetch_filing_text(
                cik, f["accessionNumber"], f["primaryDocument"]
            )
        except Exception:
            continue
        risk_text = text_signals.extract_section(full_text, text_signals.ITEM_1A_RE)
        mda_text = text_signals.extract_section(full_text, text_signals.ITEM_7_RE)

        tone = text_signals.tone_scores(risk_text + " " + mda_text)
        risk_sim = text_signals.similarity(prev_risk_text, risk_text) if prev_risk_text else float("nan")
        mda_sim = text_signals.similarity(prev_mda_text, mda_text) if prev_mda_text else float("nan")

        fy_end = pd.Timestamp(f["reportDate"])
        rows[fy_end] = {
            **tone,
            "risk_factors_similarity": risk_sim,
            "mda_similarity": mda_sim,
        }
        prev_risk_text, prev_mda_text = risk_text, mda_text

    out = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    out.index.name = "fiscal_year_end"
    return out


def _zscore(series: pd.Series) -> pd.Series:
    std = series.std()
    if not std or pd.isna(std):
        return series * 0.0
    return (series - series.mean()) / std


def composite_risk_score(fin: pd.DataFrame, text: pd.DataFrame) -> pd.DataFrame:
    """Merge the two signal sets and produce a single 0-100 composite
    risk score per fiscal year via z-scored, equal-weighted components.
    Purely a teaching heuristic -- not a calibrated probability.
    """
    merged = fin.join(text, how="outer")

    components = pd.DataFrame(index=merged.index)
    components["m_score_z"] = _zscore(merged["m_score"])
    components["negative_tone_z"] = _zscore(merged["negative_per_1k"])
    components["uncertainty_z"] = _zscore(merged["uncertainty_per_1k"])
    components["litigious_z"] = _zscore(merged["litigious_per_1k"])
    # a big *drop* in similarity (rewrite) is the risk signal, so negate it
    components["risk_rewrite_z"] = _zscore(-merged["risk_factors_similarity"])
    components["mda_rewrite_z"] = _zscore(-merged["mda_similarity"])

    raw = components.mean(axis=1, skipna=True)
    # squash to 0-100 with a logistic, centered on this company's own
    # history (there's no cross-company calibration data here, so the
    # score ranks years within a company, not companies against each other)
    merged["composite_risk_0_100"] = 100 / (1 + np.exp(-(raw - raw.mean())))
    merged["component_scores"] = components.to_dict(orient="index")

    t2 = spc.hotelling_t2(merged)
    merged = merged.join(t2)
    return merged
