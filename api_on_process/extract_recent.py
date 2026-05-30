# extract_recent.py
# ─────────────────────────────────────────────────────────────────────────────
# Pulls RECENT data (from 2026-03-19 onwards) from the Edutizim API.
# This is the live-update version of extract.py — it does NOT re-fetch
# historical data from 2020, only the data since the last known snapshot.
#
# Use this before running the live pipeline to get fresh active-student data.
#
# How to run:
#   python3 api_on_process/extract_recent.py
# ─────────────────────────────────────────────────────────────────────────────

import sys
import os
import json
import csv
import logging
import time
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

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

# ── API config ────────────────────────────────────────────────────────────────
BASE_URL     = "https://backend.edutizim.uz/external-api"
API_KEY      = os.getenv("API_KEY",      "dev:07cacc808ee6a6dd9adce3344ff9bf20")
ORGANIZATION = os.getenv("ORGANIZATION", "registan")
BRANCH       = os.getenv("BRANCH",       "6266d9e35bbdd74734fddadd")
API_PHONE    = os.getenv("API_PHONE",    "")
API_PASSWORD = os.getenv("API_PASSWORD", "")
PAGE_SIZE    = int(os.getenv("API_PAGE_SIZE", "200"))

# ── Key difference from extract.py: only fetch from March 19, 2026 ────────────
RECENT_FROM = "2026-03-19T00:00:00.000Z"

ENDPOINTS = [
    # static lookup tables — always fetch in full (small, fast)
    # optional=True means: if this endpoint fails or returns 0 rows, warn but continue.
    # The pipeline will use whatever cleaned file already exists on disk.
    dict(name="courses",          api_path="courses",          output=COURSES,          use_dates=False, optional=False),
    dict(name="branches",         api_path="branches",         output=BRANCHES,         use_dates=False, optional=False),
    dict(name="levels",           api_path="levels",           output=LEVELS,           use_dates=False, optional=True),
    # users API requires a date filter to return data (returns 0 without it — API quirk).
    # We fetch from RECENT_FROM (2026-03-19); historical users are already in data/interim/.
    dict(name="users",            api_path="users",            output=USERS,            use_dates=True,  optional=True),

    # students — fetch since March 19 for new registrations.
    # Name lookup in predict_live.py uses raw JSON + orders.studentInfo instead,
    # so this endpoint failing is non-fatal (100% names resolved without it).
    dict(name="students",         api_path="students",         output=STUDENTS,         use_dates=True,  optional=True),
    dict(name="groups",           api_path="groups",           output=GROUPS,           use_dates=True,  optional=False),
    dict(name="orders",           api_path="orders",           output=ORDERS,           use_dates=True,  optional=False),
    dict(name="studentgroups",    api_path="student-groups",   output=STUDENTGROUPS,    use_dates=True,  optional=False),
    dict(name="student-teachers", api_path="student-teachers", output=STUDENTTEACHERS,  use_dates=False, optional=True),
    dict(name="attendance",       api_path="attendances",      output=ATTENDANCE,       use_dates=True,  optional=False),
    dict(name="transactions",     api_path="transactions",     output=TRANSACTIONS,     use_dates=True,  optional=False),
    dict(name="grouphistories",   api_path="group-histories",  output=GROUPHISTORIES,   use_dates=True,  optional=False),
    dict(name="studenthistories", api_path="student-histories",output=STUDENTHISTORIES, use_dates=True,  optional=False),
    dict(name="lessons",          api_path="lessons",          output=LESSONS,          use_dates=True,  optional=False),
]


def login() -> str:
    url = f"{BASE_URL}/sign-with-password"
    headers = {"Content-Type": "application/json", "apiKey": API_KEY}
    body = {"phoneNumber": API_PHONE, "password": API_PASSWORD}
    log.info("Logging in to obtain a Bearer token ...")
    resp = requests.post(url, headers=headers, json=body, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    token = (data.get("data", {}).get("token") or data.get("data", {}).get("accessToken")
             or data.get("token") or data.get("accessToken"))
    if not token:
        raise RuntimeError(f"Login succeeded but no token found. Response: {json.dumps(data)[:300]}")
    log.info("Token obtained.")
    return token


def get_token() -> str:
    token = os.getenv("API_TOKEN", "")
    if token:
        return token
    if not API_PHONE or not API_PASSWORD:
        raise EnvironmentError(
            "API_TOKEN not set in .env and API_PHONE/API_PASSWORD also missing."
        )
    return login()


def make_headers(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "apiKey":        API_KEY,
        "organization":  ORGANIZATION,
        "branch":        BRANCH,
        "Content-Type":  "application/json",
    }


def fetch_all(api_path: str, headers: dict, use_dates: bool, from_date: str = RECENT_FROM) -> list:
    url         = f"{BASE_URL}/raw-data/{api_path}"
    to_date     = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    all_records = []
    page        = 1

    while True:
        body: dict = {"page": page, "limit": PAGE_SIZE}
        if use_dates:
            body["fromDate"] = from_date
            body["toDate"]   = to_date

        log.info(f"    page {page:>4}  ({len(all_records):>7,} fetched so far) ...")

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
        docs  = data_block.get("docs") or data_block.get("data") or []
        total = data_block.get("total", 0)

        if not docs:
            break

        all_records.extend(docs)
        log.info(f"    {len(all_records):>7,} / {total:,}")

        if len(all_records) >= total or len(docs) < PAGE_SIZE:
            break

        page += 1
        time.sleep(0.1)

    return all_records


def flatten(record: dict, prefix: str = "") -> dict:
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    flat_records = [flatten(r) for r in records]
    all_keys = list(dict.fromkeys(k for r in flat_records for k in r))
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(flat_records)
    size_kb = output_path.stat().st_size // 1024
    log.info(f"  Saved {output_path.name}  ({len(records):,} rows · {size_kb:,} KB)")


def save_json(records: list, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, default=str)
    size_kb = output_path.stat().st_size // 1024
    log.info(f"  Saved {output_path.name}  ({len(records):,} records · {size_kb:,} KB)")


def save_output(records: list, output_path: Path):
    if output_path.suffix == ".json":
        save_json(records, output_path)
    else:
        save_csv(records, output_path)


def run():
    log.info("=" * 60)
    log.info("EXTRACT (RECENT)  —  Registan API")
    log.info(f"  Fetching data from : {RECENT_FROM}")
    log.info(f"  To                 : now")
    log.info(f"  Base URL           : {BASE_URL}")
    log.info("=" * 60)

    token   = get_token()
    headers = make_headers(token)
    failed  = []

    failed_required = []
    failed_optional = []

    for ep in ENDPOINTS:
        name      = ep["name"]
        api_path  = ep["api_path"]
        output    = ep["output"]
        use_dates = ep["use_dates"]
        optional  = ep.get("optional", False)

        log.info(f"\n{'─' * 55}")
        log.info(f"  '{name}'  →  POST /raw-data/{api_path}"
                 + ("  [optional]" if optional else ""))
        log.info(f"{'─' * 55}")

        try:
            from_date = ep.get("from_date", RECENT_FROM)
            records = fetch_all(api_path, headers, use_dates, from_date=from_date)
            if not records:
                if optional:
                    log.warning(f"  No records returned for '{name}' — optional, skipping.")
                    failed_optional.append(name)
                else:
                    log.warning(f"  No records returned for '{name}' — skipping save.")
                    failed_required.append(name)
                continue
            save_output(records, output)
        except Exception as exc:
            log.error(f"  FAILED '{name}': {exc}")
            if optional:
                failed_optional.append(name)
            else:
                failed_required.append(name)

    log.info("\n" + "=" * 60)
    total_failed = len(failed_required) + len(failed_optional)
    succeeded    = len(ENDPOINTS) - total_failed
    log.info(f"EXTRACT COMPLETE  ({succeeded}/{len(ENDPOINTS)} collections)")
    if failed_optional:
        log.warning(f"  Optional (skipped, using existing files): {', '.join(failed_optional)}")
    if failed_required:
        log.warning(f"  Required (FATAL): {', '.join(failed_required)}")
        raise RuntimeError(f"Extraction failed for required endpoints: {failed_required}")
    log.info("  All required raw files written to data/raw/")
    log.info("=" * 60)


if __name__ == "__main__":
    run()
