"""Thin client for SEC EDGAR's public data APIs.

SEC requires every request to carry an identifying User-Agent
(name + contact email) or it will start throttling/blocking you.
Set SEC_EDGAR_USER_AGENT in a .env file before running anything, e.g.:

    SEC_EDGAR_USER_AGENT="Jane Student jane.student@example.com"
"""
import os
import time
import re
from functools import lru_cache
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

USER_AGENT = os.environ.get(
    "SEC_EDGAR_USER_AGENT", "SEC-Red-Flag-Detector (set SEC_EDGAR_USER_AGENT in .env)"
)
HEADERS = {"User-Agent": USER_AGENT}

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

_last_request_time = 0.0


def _throttled_get(url: str) -> requests.Response:
    """SEC asks for <=10 req/sec; we stay well under that."""
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < 0.2:
        time.sleep(0.2 - elapsed)
    resp = requests.get(url, headers=HEADERS, timeout=30)
    _last_request_time = time.time()
    resp.raise_for_status()
    return resp


@lru_cache(maxsize=1)
def _ticker_map() -> dict:
    data = _throttled_get(TICKERS_URL).json()
    return {row["ticker"].upper(): str(row["cik_str"]).zfill(10) for row in data.values()}


def cik_for_ticker(ticker: str) -> str:
    mapping = _ticker_map()
    cik = mapping.get(ticker.upper())
    if not cik:
        raise ValueError(f"Unknown ticker: {ticker}")
    return cik


def get_submissions(cik: str) -> dict:
    return _throttled_get(SUBMISSIONS_URL.format(cik=cik)).json()


def get_company_facts(cik: str) -> dict:
    return _throttled_get(COMPANYFACTS_URL.format(cik=cik)).json()


def _filings_from_block(block: dict, form_type: str) -> list[dict]:
    n = len(block["form"])
    out = []
    for i in range(n):
        if block["form"][i] == form_type:
            out.append(
                {
                    "accessionNumber": block["accessionNumber"][i],
                    "filingDate": block["filingDate"][i],
                    "reportDate": block["reportDate"][i],
                    "primaryDocument": block["primaryDocument"][i],
                }
            )
    return out


def list_filings(cik: str, form_type: str = "10-K") -> list[dict]:
    """Return all filings of a given form type, newest first. The
    submissions endpoint only inlines the most recent ~1000 filings;
    older ones live in separate paginated JSON files listed under
    filings.files, which we fetch too so multi-year history works for
    long-tenured companies.
    """
    subs = get_submissions(cik)
    out = _filings_from_block(subs["filings"]["recent"], form_type)
    for extra in subs["filings"].get("files", []):
        extra_data = _throttled_get(f"https://data.sec.gov/submissions/{extra['name']}").json()
        out.extend(_filings_from_block(extra_data, form_type))
    out.sort(key=lambda f: f["filingDate"], reverse=True)
    return out


def filing_document_url(cik: str, accession_number: str, primary_document: str) -> str:
    cik_int = str(int(cik))
    accession_nodash = accession_number.replace("-", "")
    return (
        f"https://www.sec.gov/Archives/edgar/data/{cik_int}/"
        f"{accession_nodash}/{primary_document}"
    )


def fetch_filing_text(cik: str, accession_number: str, primary_document: str) -> str:
    import warnings

    from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

    url = filing_document_url(cik, accession_number, primary_document)
    html = _throttled_get(url).text
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)
        soup = BeautifulSoup(html, "lxml")
    text = soup.get_text(separator="\n")
    text = re.sub(r"\n{2,}", "\n", text)
    return text
