# test_endpoints.py
# Run: python3 api_on_process/test_endpoints.py
# Tests the users and student-teachers endpoints to see what the API actually returns.

import sys, os, json, requests
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

BASE_URL     = "https://backend.edutizim.uz/external-api"
def _require(name):
    val = os.getenv(name, "").strip()
    if not val:
        raise EnvironmentError(f"{name} is not set. Add it to .env or GitHub Secrets.")
    return val

API_KEY      = _require("API_KEY")
ORGANIZATION = _require("ORGANIZATION")
BRANCH       = _require("BRANCH")
TOKEN        = os.getenv("API_TOKEN", "")

HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "apiKey":        API_KEY,
    "organization":  ORGANIZATION,
    "branch":        BRANCH,
    "Content-Type":  "application/json",
}

RECENT_FROM = "2026-03-19T00:00:00.000Z"
TO_DATE     = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")

tests = [
    {
        "name":      "users — no filters",
        "api_path":  "users",
        "body":      {"page": 1, "limit": 5},
        "extra_headers": {},
    },
    {
        "name":      "users — with date filter",
        "api_path":  "users",
        "body":      {"page": 1, "limit": 5, "fromDate": "2020-01-01T00:00:00.000Z", "toDate": TO_DATE},
        "extra_headers": {},
    },
    {
        "name":      "users — NO branch header",
        "api_path":  "users",
        "body":      {"page": 1, "limit": 5},
        "extra_headers": {"branch": ""},   # send without branch
    },
]

for t in tests:
    url = f"{BASE_URL}/raw-data/{t['api_path']}"
    headers = {**HEADERS, **t.get("extra_headers", {})}
    # remove branch header entirely if empty
    if "branch" in headers and headers["branch"] == "":
        del headers["branch"]

    print(f"\n{'='*60}")
    print(f"  Endpoint : {t['name']}")
    print(f"  URL      : {url}")
    print(f"  Body     : {json.dumps(t['body'])}")
    print(f"{'='*60}")
    try:
        resp = requests.post(url, headers=headers, json=t["body"], timeout=30)
        print(f"  HTTP status : {resp.status_code}")
        try:
            data = resp.json()
            block = data.get("data", {})
            docs  = block.get("docs") or block.get("data") or []
            total = block.get("total", "?")
            print(f"  total       : {total}")
            print(f"  docs count  : {len(docs)}")
            if docs:
                print(f"  First record keys: {list(docs[0].keys())[:10]}")
                print(f"  Sample record    : {json.dumps(docs[0], ensure_ascii=False, default=str)[:300]}")
            else:
                print(f"  Full response:")
                print("  " + json.dumps(data, ensure_ascii=False)[:500])
        except Exception:
            print(f"  Raw response (not JSON): {resp.text[:300]}")
    except Exception as e:
        print(f"  ERROR: {e}")
