"""Pull line items out of SEC's XBRL 'company facts' JSON into a clean
one-row-per-fiscal-year table. Different companies tag the same concept
differently (e.g. Revenues vs RevenueFromContractWithCustomerExcludingAssessedTax),
so each line item lists fallback tags in priority order.
"""
import pandas as pd

LINE_ITEMS = {
    "revenue": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
    ],
    "cogs": [
        "CostOfGoodsAndServicesSold",
        "CostOfRevenue",
        "CostOfGoodsSold",
    ],
    "receivables": [
        "AccountsReceivableNetCurrent",
        "ReceivablesNetCurrent",
    ],
    "current_assets": ["AssetsCurrent"],
    "total_assets": ["Assets"],
    "ppe_net": ["PropertyPlantAndEquipmentNet"],
    "depreciation": [
        "DepreciationDepletionAndAmortization",
        "DepreciationAndAmortization",
        "Depreciation",
    ],
    "sga_expense": [
        "SellingGeneralAndAdministrativeExpense",
        "GeneralAndAdministrativeExpense",
    ],
    "total_liabilities": ["Liabilities"],
    "stockholders_equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "cash_from_ops": [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ],
}


def _annual_facts_for_tag(company_facts: dict, tag: str, valid_ends: set[str] | None) -> list[dict]:
    """Return USD facts for `tag` that look like a full fiscal year
    (duration 300-400 days) reported on a 10-K.

    A 10-K sometimes discloses *other* periods' figures too -- e.g. a
    restatement filing's quarterly reconciliation footnote -- all tagged
    with the same form/fy/fp metadata as the real annual figure, so those
    alone can't distinguish them. When `valid_ends` (the company's actual
    10-K fiscal-year-end dates) is supplied, we additionally require the
    fact's own 'end' to be one of them.
    """
    try:
        units = company_facts["facts"]["us-gaap"][tag]["units"]["USD"]
    except KeyError:
        return []
    out = []
    for fact in units:
        if fact.get("form") != "10-K":
            continue
        start, end = fact.get("start"), fact.get("end")
        if start and end:
            days = (pd.Timestamp(end) - pd.Timestamp(start)).days
            if not (300 <= days <= 400):
                continue
        if valid_ends is not None and end not in valid_ends:
            continue
        out.append(fact)
    return out


def _instant_facts_for_tag(company_facts: dict, tag: str, valid_ends: set[str] | None) -> list[dict]:
    """Balance-sheet items have no 'start', just an 'end' instant. See
    _annual_facts_for_tag for why `valid_ends` filtering matters here too --
    it's actually more important for instant facts, since a 10-K with a
    quarterly-restatement footnote reports a same-shaped 'no start, has
    end' fact for every quarter, not just fiscal year end.
    """
    try:
        units = company_facts["facts"]["us-gaap"][tag]["units"]["USD"]
    except KeyError:
        return []
    out = []
    for f in units:
        if f.get("form") != "10-K" or f.get("start"):
            continue
        if valid_ends is not None and f.get("end") not in valid_ends:
            continue
        out.append(f)
    return out


DURATION_ITEMS = {"revenue", "cogs", "depreciation", "sga_expense", "net_income", "cash_from_ops"}


def build_annual_dataframe(company_facts: dict, valid_fiscal_year_ends: list[str] | None = None) -> pd.DataFrame:
    """One row per fiscal year end, one column per line item, fallback-resolved.

    `valid_fiscal_year_ends`: the company's real 10-K report dates (e.g. from
    edgar_client.list_filings), strongly recommended -- see
    _annual_facts_for_tag / _instant_facts_for_tag for why. Without it,
    filings that disclose other periods' figures (quarterly restatement
    footnotes, etc.) can silently corrupt the annual series.
    """
    rows: dict[str, dict] = {}
    valid_ends = set(valid_fiscal_year_ends) if valid_fiscal_year_ends is not None else None

    for item, tags in LINE_ITEMS.items():
        is_duration = item in DURATION_ITEMS
        for tag in tags:
            facts = (
                _annual_facts_for_tag(company_facts, tag, valid_ends)
                if is_duration
                else _instant_facts_for_tag(company_facts, tag, valid_ends)
            )
            for fact in facts:
                fy_end = fact["end"]
                rows.setdefault(fy_end, {})
                # first tag in the fallback list to report a value for this
                # fiscal year wins; don't let a later fallback overwrite it
                rows[fy_end].setdefault(item, fact["val"])

    df = pd.DataFrame.from_dict(rows, orient="index")
    # guarantee every line item is a column, even if no tag matched for any
    # year, so downstream ratio math gets a well-formed (if all-NaN) Series
    # via KeyError-free `df["revenue"]` instead of crashing outright
    df = df.reindex(columns=list(LINE_ITEMS.keys()))

    # some filers (Nike, Fluor, Granite Construction, ...) never tag a
    # single roll-up "Liabilities" total -- derive it from the accounting
    # identity Assets = Liabilities + Equity wherever it's missing
    derived_liabilities = df["total_assets"] - df["stockholders_equity"]
    df["total_liabilities"] = df["total_liabilities"].fillna(derived_liabilities)

    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df.index.name = "fiscal_year_end"
    return df
