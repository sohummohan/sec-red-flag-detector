import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Streamlit Community Cloud has no .env file -- secrets are configured in
# the dashboard and surfaced via st.secrets instead. Locally, .env (loaded
# by edgar_client) already covers this, so this is a no-op there.
if "SEC_EDGAR_USER_AGENT" in st.secrets:
    os.environ["SEC_EDGAR_USER_AGENT"] = st.secrets["SEC_EDGAR_USER_AGENT"]

import beneish
import edgar_client
import pipeline

st.set_page_config(page_title="SEC Filing Red-Flag Detector", layout="wide")

st.title("SEC Filing Red-Flag Detector")
st.caption(
    "Combines the Beneish M-Score (the earnings-manipulation model that flagged Enron) "
    "with year-over-year tone and rewrite analysis of 10-K narrative sections, "
    "pulled live from SEC EDGAR."
)

with st.sidebar:
    st.header("Company")
    ticker = st.text_input("Ticker", value="UAA").strip().upper()
    max_filings = st.slider("Years of 10-Ks to analyze", 3, 15, 10)
    run = st.button("Analyze", type="primary")
    st.divider()
    st.caption(
        "Known validation case: **UAA fiscal 2015** — the SEC's 2021 settlement found "
        "Under Armour failed to disclose that it pulled forward $408M in sales over "
        "6 quarters starting Q3 2015 (a disclosure failure, not an alleged GAAP "
        "violation) — this year scores as the single highest M-Score in the company's "
        "filing history. See the full validation study in the README for more cases."
    )
    st.caption(
        "This is a research/educational tool, not investment advice. Scores are heuristic "
        "and relative to each company's own history."
    )

if not run:
    st.info("Enter a ticker and click **Analyze** to pull its SEC filings and compute risk signals.")
    st.stop()

try:
    with st.spinner(f"Resolving {ticker} on EDGAR..."):
        cik = edgar_client.cik_for_ticker(ticker)
except ValueError:
    st.error(f"Couldn't find ticker '{ticker}' on SEC EDGAR. Check the symbol and try again.")
    st.stop()

with st.spinner("Pulling structured financials (XBRL) and computing Beneish M-Score..."):
    try:
        fin = pipeline.financial_signals(cik)
    except Exception as e:
        st.error(f"Couldn't compute financial signals: {e}")
        st.stop()

with st.spinner(f"Fetching and parsing up to {max_filings} 10-Ks (this can take a minute)..."):
    try:
        text = pipeline.textual_signals(cik, max_filings=max_filings)
    except Exception as e:
        st.error(f"Couldn't compute text signals: {e}")
        st.stop()

merged = pipeline.composite_risk_score(fin, text)
merged = merged.dropna(subset=["composite_risk_0_100"])

if merged.empty:
    st.warning("Not enough overlapping financial + text data to score this company.")
    st.stop()

latest = merged.iloc[-1]
worst = merged.loc[merged["composite_risk_0_100"].idxmax()]

col1, col2, col3 = st.columns(3)
col1.metric("Latest fiscal year", merged.index[-1].strftime("%Y-%m-%d"))
col2.metric("Latest composite risk (0-100)", f"{latest['composite_risk_0_100']:.0f}")
col3.metric(
    "Highest-risk year on record",
    worst.name.strftime("%Y"),
    f"score {worst['composite_risk_0_100']:.0f}",
)

st.subheader("Composite risk score over time")
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=merged.index,
        y=merged["composite_risk_0_100"],
        mode="lines+markers",
        name="Composite risk",
        line=dict(color="#d62728", width=3),
    )
)
fig.add_hline(y=70, line_dash="dot", line_color="gray", annotation_text="elevated")
fig.update_layout(yaxis_title="Risk score (0-100, relative to own history)", xaxis_title="Fiscal year")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Signal breakdown")
tab1, tab2, tab3 = st.tabs(
    ["Financial: Beneish M-Score", "Textual: tone & rewrite analysis", "Multivariate: Hotelling T² (SPC)"]
)

with tab1:
    st.markdown(
        f"M-Score > **{beneish.THRESHOLD}** is the conventional threshold for a likely earnings manipulator."
    )
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=merged.index, y=merged["m_score"], mode="lines+markers", name="M-Score"))
    fig2.add_hline(y=beneish.THRESHOLD, line_dash="dot", line_color="red", annotation_text="threshold")
    fig2.update_layout(yaxis_title="M-Score", xaxis_title="Fiscal year")
    st.plotly_chart(fig2, use_container_width=True)
    st.dataframe(
        merged[["DSRI", "GMI", "AQI", "SGI", "DEPI", "SGAI", "LVGI", "TATA", "m_score", "flagged"]].round(3)
    )

with tab2:
    fig3 = go.Figure()
    for col, label in [
        ("negative_per_1k", "Negative language"),
        ("uncertainty_per_1k", "Uncertainty language"),
        ("litigious_per_1k", "Litigious language"),
    ]:
        fig3.add_trace(go.Scatter(x=merged.index, y=merged[col], mode="lines+markers", name=label))
    fig3.update_layout(yaxis_title="Occurrences per 1,000 words", xaxis_title="Fiscal year")
    st.plotly_chart(fig3, use_container_width=True)

    fig4 = go.Figure()
    fig4.add_trace(
        go.Scatter(x=merged.index, y=merged["risk_factors_similarity"], mode="lines+markers", name="Risk Factors similarity")
    )
    fig4.add_trace(
        go.Scatter(x=merged.index, y=merged["mda_similarity"], mode="lines+markers", name="MD&A similarity")
    )
    fig4.update_layout(
        yaxis_title="Cosine similarity vs. prior year (1.0 = unchanged)",
        xaxis_title="Fiscal year",
        yaxis_range=[0, 1],
    )
    st.plotly_chart(fig4, use_container_width=True)
    st.caption("A sharp drop means the section was substantially rewritten year-over-year -- sometimes benign, sometimes worth a closer read.")

with tab3:
    st.markdown(
        "**Hotelling's T²** is the standard industrial-engineering technique for monitoring a "
        "multivariate process (taught e.g. in UC Berkeley IEOR 165, *Engineering Statistics, "
        "Quality Control, and Forecasting*) -- normally used to watch a manufacturing line for "
        "drift. Here the '8 sensors' are the Beneish ratios, and the 'process' is a company's own "
        "reported financials. Unlike the composite score above (which treats each ratio "
        "independently), T² accounts for how the ratios normally move *together*, so it won't "
        "overreact to two correlated ratios moving in their usual lockstep. A year crossing the "
        "UCL (upper control limit, derived from the F-distribution at 95% confidence) is "
        "statistically out of control relative to the company's own history."
    )
    t2_data = merged.dropna(subset=["t2", "ucl"])
    if t2_data.empty:
        st.info("Not enough years of clean financial data to fit a multivariate control chart for this company (need at least ~11 years of complete Beneish ratios).")
    else:
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(x=t2_data.index, y=t2_data["t2"], mode="lines+markers", name="T² statistic"))
        fig5.add_trace(go.Scatter(x=t2_data.index, y=t2_data["ucl"], mode="lines", name="UCL (95%)", line=dict(dash="dot", color="red")))
        fig5.update_layout(yaxis_title="Hotelling T²", xaxis_title="Fiscal year")
        st.plotly_chart(fig5, use_container_width=True)
        st.dataframe(t2_data[["t2", "ucl", "out_of_control"]].round(2))

st.divider()
st.caption("Data: SEC EDGAR (XBRL company facts + 10-K filings). Not investment advice.")
