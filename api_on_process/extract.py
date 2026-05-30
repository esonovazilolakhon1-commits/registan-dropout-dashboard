# extract.py — pulls all collections from Registan's REST API and saves to data/raw/
#
# API details (from testing in test/ scripts):
#   Base URL : https://backend.edutizim.uz/external-api
#   Endpoint : POST /raw-data/{collection}
#   Headers  : Authorization, apiKey, organization, branch, Content-Type
#   Body     : {"page": N, "limit": 200, "fromDate": "...", "toDate": "..."}
#   Response : {"data": {"total": N, "docs": [...]}} or {"data": {"total": N, "data": [...]}}
#
# Token management: the JWT token has a ~1 year expiry. Store it in .env as API_TOKEN.
# If missing or expired, the script logs in automatically using API_PHONE + API_PASSWORD.
#
# How to run:
#   python3 src/pipeline/extract.py

import sys
import os
import json
import csv
import logging
import time
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv()

import requests

from src.config import (
    RAW,
    STUDENTS, GROUPS, COURSES, ORDERS,
    STUDENTGROUPS, GROUPHISTORIES,
    STUDENTHISTORIES, LESSONS, ATTENDANCE, TRANSACTIONS,
    BRANCHES, LEVELS, USERS, STUDENTTEACHERS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# API configuration (values come from .env)
BASE_URL     = "https://backend.edutizim.uz/external-api"
API_KEY      = os.getenv("API_KEY",      "dev:07cacc808ee6a6dd9adce3344ff9bf20")
ORGANIZATION = os.getenv("ORGANIZATION", "registan")
BRANCH       = os.getenv("BRANCH",       "6266d9e35bbdd74734fddadd")
API_PHONE    = os.getenv("API_PHONE",    "")
API_PASSWORD = os.getenv("API_PASSWORD", "")
PAGE_SIZE    = int(os.getenv("API_PAGE_SIZE", "200"))

FULL_HISTORY_FROM = "2020-01-01T00:00:00.000Z"

ENDPOINTS = [
    # static lookup tables (no date filter needed)
    dict(name="courses",          api_path="courses",          output=COURSES,          use_dates=False),
    dict(name="branches",         api_path="branches",         output=BRANCHES,         use_dates=False),

    # student and group master records
    dict(name="students",         api_path="students",         output=STUDENTS,         use_dates=True),
    dict(name="groups",           api_path="groups",           output=GROUPS,           use_dates=True),
    dict(name="orders",           api_path="orders",           output=ORDERS,           use_dates=True),

    # enrolment tables
    dict(name="studentgroups",    api_path="student-groups",   output=STUDENTGROUPS,    use_dates=True),

    # event and history tables (largest collections)
    dict(name="attendance",       api_path="attendances",      output=ATTENDANCE,       use_dates=True),
    dict(name="transactions",     api_path="transactions",     output=TRANSACTIONS,     use_dates=True),
    dict(name="grouphistories",   api_path="group-histories",  output=GROUPHISTORIES,   use_dates=True),
    dict(name="studenthistories", api_path="student-histories",output=STUDENTHISTORIES, use_dates=True),
    dict(name="lessons",          api_path="lessons",          output=LESSONS,          use_dates=True),

    # Previously returned 404 — now confirmed working (HTTP 200)
    dict(name="levels",           api_path="levels",           output=LEVELS,           use_dates=False),
    dict(name="users",            api_path="users",            output=USERS,            use_dates=False),
    dict(name="student-teachers", api_path="student-teachers", output=STUDENTTEACHERS,  use_dates=True),
]


def login() -> str:
    """POST /sign-with-password to get a Bearer token. Save the result to .env as API_TOKEN."""
    url = f"{BASE_URL}/sign-with-password"
    headers = {
        "Content-Type": "application/json",
        "apiKey": API_KEY,
    }
    body = {
        "phoneNumber": API_PHONE,
        "password":    API_PASSWORD,
    }
    log.info("  Logging in to obtain a Bearer token ...")
    resp = requests.post(url, headers=headers, json=body, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    token = (
        data.get("data", {}).get("token") or
        data.get("data", {}).get("accessToken") or
        data.get("token") or
        data.get("accessToken")
    )
    if not token:
        raise RuntimeError(
            f"Login succeeded (HTTP 200) but no token found in response.\n"
            f"Full response: {json.dumps(data, indent=2)}"
        )

    log.info("  Token obtained.")
    log.info(f"  Add this to your .env file:  API_TOKEN={token}")
    return token


def get_token() -> str:
    """Return a valid Bearer token, logging in if needed."""
    token = os.getenv("API_TOKEN", "")
    if token:
        return token
    if not API_PHONE or not API_PASSWORD:
        raise EnvironmentError(
            "API_TOKEN is not set in .env and API_PHONE/API_PASSWORD are also missing.\n"
            "Either paste your token directly (API_TOKEN=...) or add your phone and password."
        )
    return login()


def make_headers(token: str) -> dict:
    """Build the headers required by every request to this API."""
    return {
        "Authorization": f"Bearer {token}",
        "apiKey":        API_KEY,
        "organization":  ORGANIZATION,
        "branch":        BRANCH,
        "Content-Type":  "application/json",
    }


def fetch_all(api_path: str, headers: dict, use_dates: bool) -> list:
    """Fetch all records from one endpoint, handling pagination automatically."""
    url         = f"{BASE_URL}/raw-data/{api_path}"
    to_date     = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    all_records = []
    page        = 1

    while True:
        body: dict = {"page": page, "limit": PAGE_SIZE}
        if use_dates:
            body["fromDate"] = FULL_HISTORY_FROM
            body["toDate"]   = to_date

        log.info(f"    page {page:>4}  ({len(all_records):>7,} fetched so far) ...")

        # retry up to 5 times with escalating back-off
        for attempt in range(5):
            try:
                resp = requests.post(url, headers=headers, json=body, timeout=120)
                resp.raise_for_status()
                break
            except requests.exceptions.RequestException as exc:
                wait = 10 * (attempt + 1)
                log.warning(f"    [retry {attempt+1}/5] {exc.__class__.__name__} — waiting {wait}s")
                time.sleep(wait)
        else:
            raise RuntimeError(f"All 5 retries failed for {api_path} page {page}")

        data_block = resp.json().get("data", {})

        # the API uses 'docs' or 'data' as the key for the records array
        docs  = data_block.get("docs") or data_block.get("data") or []
        total = data_block.get("total", 0)

        if not docs:
            log.info(f"    Empty page — done.")
            break

        all_records.extend(docs)
        log.info(f"    {len(all_records):>7,} / {total:,}")

        if len(all_records) >= total or len(docs) < PAGE_SIZE:
            break

        page += 1
        time.sleep(0.1)  # be polite — avoid rate-limiting

    return all_records


def flatten(record: dict, prefix: str = "") -> dict:
    """
    Recursively flatten a nested dict into a single-level dict for CSV output.
    Lists are JSON-encoded as strings so they fit in a single CSV cell.
    """
    out: dict = {}
    for key, value in record.items():
        full_key = key if not prefix else f"{prefix}.{key}"
        if isinstance(value, dict):
            out.update(flatten(value, full_key))
        elif isinstance(value, list):
            out[full_key] = json.dumps(value, ensure_ascii=False)
        else:
            out[full_key] = value
    return out


def save_csv(records: list, output_path: Path):
    """Flatten each record and write to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    flat_records = [flatten(r) for r in records]
    all_keys = list(dict.fromkeys(k for r in flat_records for k in r))

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(flat_records)

    size_kb = output_path.stat().st_size // 1024
    log.info(f"  Saved {output_path.name}  "
             f"({len(records):,} rows · {len(all_keys)} cols · {size_kb:,} KB)")


def save_json(records: list, output_path: Path):
    """Save records as a JSON array — needed for ijson streaming in the clean scripts."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, default=str)
    size_kb = output_path.stat().st_size // 1024
    log.info(f"  Saved {output_path.name}  ({len(records):,} records · {size_kb:,} KB)")


def save_output(records: list, output_path: Path):
    """Route to save_json for .json files, save_csv for .csv files."""
    if output_path.suffix == ".json":
        save_json(records, output_path)
    else:
        save_csv(records, output_path)


def run():
    log.info("=" * 60)
    log.info("EXTRACT  —  Registan API  (edutizim.uz)")
    log.info(f"  Base URL     : {BASE_URL}")
    log.info(f"  Organization : {ORGANIZATION}")
    log.info(f"  Branch       : {BRANCH}")
    log.info(f"  Page size    : {PAGE_SIZE}")
    log.info(f"  History from : {FULL_HISTORY_FROM}")
    log.info("=" * 60)

    token   = get_token()
    headers = make_headers(token)

    failed = []

    for ep in ENDPOINTS:
        name      = ep["name"]
        api_path  = ep["api_path"]
        output    = ep["output"]
        use_dates = ep["use_dates"]

        log.info(f"\n{'─' * 55}")
        log.info(f"  '{name}'  →  POST /raw-data/{api_path}")
        log.info(f"{'─' * 55}")

        try:
            records = fetch_all(api_path, headers, use_dates)

            if not records:
                log.warning(f"  No records returned for '{name}' — skipping save.")
                failed.append(name)
                continue

            save_output(records, output)

        except Exception as exc:
            log.error(f"  FAILED '{name}': {exc}")
            failed.append(name)

    log.info("\n" + "=" * 60)
    succeeded = len(ENDPOINTS) - len(failed)
    log.info(f"EXTRACT COMPLETE  ({succeeded}/{len(ENDPOINTS)} collections)")

    if failed:
        log.warning(f"  Failed: {', '.join(failed)}")
        log.warning("  Fix the errors above and re-run before proceeding to clean.")
        raise RuntimeError(f"Extraction failed for: {failed}")

    log.info("  All raw files written to data/raw/")
    log.info("=" * 60)


if __name__ == "__main__":
    run()
