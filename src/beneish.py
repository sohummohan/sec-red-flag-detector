"""Beneish M-Score: an academic earnings-manipulation model (Beneish, 1999)
built from 8 ratios of year-over-year change in a company's financials.
It's the same style of model that flagged Enron before its collapse.

M-Score > -1.78 is the conventional threshold for "likely manipulator".
This is a statistical heuristic, not a verdict -- treat it as one input
into a broader risk picture, alongside the text-tone signals.
"""
import numpy as np
import pandas as pd

THRESHOLD = -1.78

WEIGHTS = {
    "DSRI": 0.920,
    "GMI": 0.528,
    "AQI": 0.404,
    "SGI": 0.892,
    "DEPI": 0.115,
    "SGAI": -0.172,
    "TATA": 4.679,
    "LVGI": -0.327,
}
INTERCEPT = -4.84


def compute_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """df: output of xbrl_utils.build_annual_dataframe, one row per fiscal
    year end, sorted ascending. Returns ratios computed for year t vs t-1,
    so the first row is always NaN.
    """
    d = df.copy()
    prev = d.shift(1)

    gross_margin = (d["revenue"] - d["cogs"]) / d["revenue"]
    gross_margin_prev = (prev["revenue"] - prev["cogs"]) / prev["revenue"]

    dep_rate = d["depreciation"] / (d["depreciation"] + d["ppe_net"])
    dep_rate_prev = prev["depreciation"] / (prev["depreciation"] + prev["ppe_net"])

    leverage = d["total_liabilities"] / d["total_assets"]
    leverage_prev = prev["total_liabilities"] / prev["total_assets"]

    asset_quality = 1 - (d["current_assets"] + d["ppe_net"]) / d["total_assets"]
    asset_quality_prev = 1 - (prev["current_assets"] + prev["ppe_net"]) / prev["total_assets"]

    ratios = pd.DataFrame(index=d.index)
    ratios["DSRI"] = (d["receivables"] / d["revenue"]) / (prev["receivables"] / prev["revenue"])
    ratios["GMI"] = gross_margin_prev / gross_margin
    ratios["AQI"] = asset_quality / asset_quality_prev
    ratios["SGI"] = d["revenue"] / prev["revenue"]
    ratios["DEPI"] = dep_rate_prev / dep_rate
    ratios["SGAI"] = (d["sga_expense"] / d["revenue"]) / (prev["sga_expense"] / prev["revenue"])
    ratios["LVGI"] = leverage / leverage_prev
    ratios["TATA"] = (d["net_income"] - d["cash_from_ops"]) / d["total_assets"]
    return ratios


def compute_m_score(df: pd.DataFrame) -> pd.DataFrame:
    ratios = compute_ratios(df)
    m_score = INTERCEPT + sum(ratios[k] * w for k, w in WEIGHTS.items())
    out = ratios.copy()
    out["m_score"] = m_score
    out["flagged"] = out["m_score"] > THRESHOLD
    return out
