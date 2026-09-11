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
        "SalesRevenueNet",
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
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "cash_from_ops": [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ],
}


def _annual_facts_for_tag(company_facts: dict, tag: str) -> list[dict]:
    """Return USD facts for `tag` that look like a full fiscal year
    (duration 300-400 days) reported on a 10-K."""
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
        out.append(fact)
    return out


def _instant_facts_for_tag(company_facts: dict, tag: str) -> list[dict]:
    """Balance-sheet items have no 'start', just an 'end' instant."""
    try:
        units = company_facts["facts"]["us-gaap"][tag]["units"]["USD"]
    except KeyError:
        return []
    return [f for f in units if f.get("form") == "10-K" and not f.get("start")]


DURATION_ITEMS = {"revenue", "cogs", "depreciation", "sga_expense", "net_income", "cash_from_ops"}


def build_annual_dataframe(company_facts: dict) -> pd.DataFrame:
    """One row per fiscal year end, one column per line item, fallback-resolved."""
    rows: dict[str, dict] = {}

    for item, tags in LINE_ITEMS.items():
        is_duration = item in DURATION_ITEMS
        for tag in tags:
            facts = (
                _annual_facts_for_tag(company_facts, tag)
                if is_duration
                else _instant_facts_for_tag(company_facts, tag)
            )
            for fact in facts:
                fy_end = fact["end"]
                rows.setdefault(fy_end, {})
                # first tag in the fallback list to report a value for this
                # fiscal year wins; don't let a later fallback overwrite it
                rows[fy_end].setdefault(item, fact["val"])

    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df.index.name = "fiscal_year_end"
    return df
