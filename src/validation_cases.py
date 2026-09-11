"""Documented SEC enforcement cases used to validate the model, plus
matched "clean" companies (same industry, no known accounting issues)
used as controls. Every fraud case cites its SEC source -- verify
against the citation before using these facts anywhere (e.g. an essay).

Note on wording: not all of these are GAAP/accounting-fraud findings.
Under Armour's 2021 settlement was specifically for a *disclosure*
failure around pulled-forward sales; the SEC explicitly did not allege
the sales themselves violated GAAP. It's included because pulling sales
forward is exactly the kind of revenue-timing manipulation the Beneish
model (via DSRI/SGI) is designed to catch, but call it what the SEC
actually said, not "accounting fraud."
"""

FRAUD_CASES = [
    {
        "ticker": "UAA",
        "name": "Under Armour",
        "flagged_fiscal_years": [2015, 2016],
        "conduct": "Undisclosed 'pulling forward' of $408M in orders across 6 quarters "
        "(Q3 2015-Q4 2016) to hide slowing demand; SEC found a disclosure failure, "
        "not a GAAP violation.",
        "citation": "https://www.sec.gov/newsroom/press-releases/2021-78",
        "settled": 2021,
    },
    {
        "ticker": "KHC",
        "name": "Kraft Heinz",
        # per the SEC's own press release: "from the last quarter of 2015
        # to the end of 2018, Kraft engaged in various types of accounting
        # misconduct" -- broader than the commonly-cited "restated 2016-2017"
        "flagged_fiscal_years": [2016, 2017, 2018],
        "conduct": "Years-long scheme faking supplier contracts to book cost savings "
        "not actually earned; $208M improperly recognized. SEC: misconduct ran "
        "Q4 2015 through end of 2018.",
        "citation": "https://www.sec.gov/newsroom/press-releases/2021-174",
        "settled": 2021,
    },
    {
        "ticker": "MDXG",
        "name": "MiMedx",
        # SEC's complaint: fraudulent revenue recognition "from 2013 to 2017"
        "flagged_fiscal_years": [2013, 2014, 2015, 2016, 2017],
        "conduct": "Premature revenue recognition via undisclosed side arrangements "
        "with distributors, 2013-2017 per SEC complaint.",
        "citation": "https://www.sec.gov/newsroom/press-releases/2019-243",
        "settled": 2019,
    },
    {
        "ticker": "GVA",
        "name": "Granite Construction",
        "flagged_fiscal_years": [2017, 2018, 2019],
        "conduct": "Manipulated profit margins and improperly deferred expected "
        "project costs to hide underperformance in one subdivision.",
        "citation": "https://www.sec.gov/newsroom/press-releases/2022-150",
        "settled": 2022,
    },
]

# same industry as each fraud case, no known SEC accounting enforcement action
CLEAN_CASES = [
    {"ticker": "NKE", "name": "Nike", "matched_to": "UAA"},
    {"ticker": "COLM", "name": "Columbia Sportswear", "matched_to": "UAA"},
    {"ticker": "GIS", "name": "General Mills", "matched_to": "KHC"},
    {"ticker": "CPB", "name": "Campbell's", "matched_to": "KHC"},
    {"ticker": "IART", "name": "Integra LifeSciences", "matched_to": "MDXG"},
    {"ticker": "ORGO", "name": "Organogenesis", "matched_to": "MDXG"},
    {"ticker": "FLR", "name": "Fluor", "matched_to": "GVA"},
    {"ticker": "ACM", "name": "AECOM", "matched_to": "GVA"},
]

ALL_CASES = [{**c, "label": "fraud"} for c in FRAUD_CASES] + [
    {**c, "label": "clean", "flagged_fiscal_years": []} for c in CLEAN_CASES
]
