"""Run the pipeline across every case in validation_cases.py and evaluate
the composite risk score as a binary classifier for "was this fiscal year
part of a documented SEC enforcement case." Pooled across all company-years
(both flagged and non-flagged years of fraud companies, plus every year of
clean companies), so it tests both sensitivity and specificity, not just
"did the model rank the bad year highest within one company."

Usage:
    python src/validation_study.py [--max-filings N] [--out results.csv]
"""
import argparse
import sys
import traceback

import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix

import edgar_client
import pipeline
from validation_cases import ALL_CASES

ELEVATED_THRESHOLD = 70


def run_case(case: dict, max_filings: int) -> pd.DataFrame | None:
    try:
        cik = edgar_client.cik_for_ticker(case["ticker"])
        fin = pipeline.financial_signals(cik)
        text = pipeline.textual_signals(cik, max_filings=max_filings)
        merged = pipeline.composite_risk_score(fin, text)
    except Exception:
        print(f"  [skip] {case['ticker']}: {traceback.format_exc(limit=1).splitlines()[-1]}")
        return None

    merged = merged.dropna(subset=["composite_risk_0_100"]).copy()
    if merged.empty:
        print(f"  [skip] {case['ticker']}: no overlapping financial+text years")
        return None

    merged["ticker"] = case["ticker"]
    merged["label"] = case["label"]
    merged["fiscal_year"] = merged.index.year
    merged["is_flagged_year"] = merged["fiscal_year"].isin(case["flagged_fiscal_years"]).astype(int)
    cols = ["ticker", "label", "fiscal_year", "composite_risk_0_100", "is_flagged_year"]
    for c in ["t2", "ucl", "t2_ratio", "out_of_control"]:
        if c not in merged.columns:
            merged[c] = float("nan")
    return merged[cols + ["t2", "ucl", "t2_ratio", "out_of_control"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-filings", type=int, default=12)
    parser.add_argument("--out", default="validation_results.csv")
    args = parser.parse_args()

    all_rows = []
    for case in ALL_CASES:
        print(f"Running {case['ticker']} ({case['label']})...")
        result = run_case(case, args.max_filings)
        if result is not None:
            all_rows.append(result)

    if not all_rows:
        print("No results produced -- check EDGAR access / SEC_EDGAR_USER_AGENT.")
        sys.exit(1)

    df = pd.concat(all_rows, ignore_index=True)
    df.to_csv(args.out, index=False)
    print(f"\nWrote {len(df)} company-year rows to {args.out}")

    y_true = df["is_flagged_year"]

    print(f"\nCompany-years analyzed: {len(df)} ({df['ticker'].nunique()} companies)")
    print(f"Flagged (positive) company-years: {y_true.sum()}")

    def evaluate(name: str, score: pd.Series, threshold: float):
        mask = score.notna()
        yt, ys = y_true[mask], score[mask]
        if yt.nunique() < 2:
            print(f"\n{name}: not enough positive/negative examples.")
            return
        auc = roc_auc_score(yt, ys)
        print(f"\n{name} -- pooled ROC-AUC: {auc:.3f}  (0.5 = random, 1.0 = perfect, n={mask.sum()})")
        y_pred = (ys >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(yt, y_pred).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")
        specificity = tn / (tn + fp) if (tn + fp) else float("nan")
        print(f"  At threshold={threshold}:")
        print(f"    Sensitivity (recall on documented fraud years): {sensitivity:.1%}  ({tp}/{tp+fn})")
        print(f"    Specificity (correctly quiet on clean years):   {specificity:.1%}  ({tn}/{tn+fp})")
        print(f"    False positives: {fp}, False negatives: {fn}")

    evaluate("Composite risk score (0-100, per-company normalized)", df["composite_risk_0_100"], ELEVATED_THRESHOLD)
    evaluate("Hotelling T-squared ratio (t2/UCL, cross-company comparable)", df["t2_ratio"], 1.0)

    print("\nPer-case detail:")
    for ticker, g in df.groupby("ticker"):
        flagged = g[g["is_flagged_year"] == 1]
        max_t2 = g["t2_ratio"].max()
        t2_str = f"max T2/UCL {max_t2:.2f}" if pd.notna(max_t2) else "T2 n/a"
        if flagged.empty:
            print(f"  {ticker:6s} (clean)  max score {g['composite_risk_0_100'].max():5.1f}  {t2_str}")
        else:
            rank = (g["composite_risk_0_100"] > flagged["composite_risk_0_100"].max()).sum() + 1
            flagged_t2 = flagged["t2_ratio"].max()
            flagged_t2_str = f"flagged-year T2/UCL {flagged_t2:.2f}" if pd.notna(flagged_t2) else "T2 n/a"
            print(
                f"  {ticker:6s} (fraud)  flagged-year max score "
                f"{flagged['composite_risk_0_100'].max():5.1f}  "
                f"(rank {rank}/{len(g)} within own history)  {flagged_t2_str}"
            )


if __name__ == "__main__":
    main()
