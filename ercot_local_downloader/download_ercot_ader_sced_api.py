#!/usr/bin/env python3
"""
ERCOT ADER SCED direct API downloader.

Downloads ONLY the row-level Load Resource Data in SCED records needed for specified
ADER resource names from NP3-965-ER. It does not download the large daily disclosure ZIPs.

ERCOT Public API documentation:
  https://developer.ercot.com/applications/pubapi/
Endpoint:
  https://api.ercot.com/api/public-reports/np3-965-er/60_load_res_data_in_sced

Row-level Public API data for NP3-965-ER is available from the Public API beta
activation (2023-12-11) onward. Earlier data must be obtained from historic archives.
"""

from __future__ import annotations

import argparse
import csv
import getpass
import json
import math
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path

API_BASE = "https://api.ercot.com/api/public-reports"
ENDPOINT = "/np3-965-er/60_load_res_data_in_sced"
AUTH_URL = (
    "https://ercotb2c.b2clogin.com/ercotb2c.onmicrosoft.com/"
    "B2C_1_PUBAPI-ROPC-FLOW/oauth2/v2.0/token"
)
CLIENT_ID = "fec253ea-0d06-4272-a5e6-b478baeecd70"
SCOPE = f"openid {CLIENT_ID} offline_access"
API_DATA_START = date(2023, 12, 11)
DEFAULT_RESOURCES = ["AR_ALD1", "BOERNE_ALD1"]
DEFAULT_PAGE_SIZE = 10000
DEFAULT_CHUNK_DAYS = 31
# ERCOT limit is 30 requests/min; >=2 seconds/request stays under the limit.
DEFAULT_SLEEP_SECONDS = 2.1

# Compact columns aimed at the actual dispatch/response time series.  Missing columns
# are simply omitted if ERCOT changes the schema.
CORE_COLUMNS = [
    "SCEDTimestamp",
    "repeatHourFlag",
    "qseName",
    "dmeName",
    "resourceName",
    "telResStatus",
    "maxPowerConsumption",
    "lowPowerConsumption",
    "realPowerConsumption",
    "rampRateUp",
    "rampRateDown",
    "basePoint",
    "ASAwardsECRS",
    "selfECRS",
    "selfRRSFFR",
    "selfRRSUFR",
]


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def iso_start(d: date) -> str:
    return f"{d.isoformat()}T00:00:00"


def iso_end_exclusive(d: date) -> str:
    # API filters are inclusive in practice; use the final second of the day.
    return f"{d.isoformat()}T23:59:59"


def chunks(start: date, end: date, days: int):
    cur = start
    while cur <= end:
        stop = min(end, cur + timedelta(days=days - 1))
        yield cur, stop
        cur = stop + timedelta(days=1)


def request_json(url: str, *, headers=None, data=None, method=None, timeout=90, binary=False):
    req = urllib.request.Request(
        url,
        headers=headers or {},
        data=data,
        method=method or ("POST" if data is not None else "GET"),
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            return raw if binary else json.loads(raw.decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} for {url}\n{body[:1200]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Network error for {url}: {exc}") from exc


def authenticate(username: str, password: str) -> str:
    payload = urllib.parse.urlencode(
        {
            "username": username,
            "password": password,
            "grant_type": "password",
            "scope": SCOPE,
            "client_id": CLIENT_ID,
            "response_type": "id_token",
        }
    ).encode("utf-8")
    response = request_json(
        AUTH_URL,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data=payload,
        method="POST",
        timeout=90,
    )
    # ERCOT documentation calls for the ID token in the Authorization header.
    token = response.get("id_token") or response.get("access_token")
    if not token:
        msg = response.get("error_description") or response.get("error") or str(response)
        raise RuntimeError(f"ERCOT authentication failed: {msg}")
    return token


class ErcotClient:
    def __init__(self, username: str, password: str, subscription_key: str, sleep_seconds: float):
        self.username = username
        self.password = password
        self.subscription_key = subscription_key
        self.sleep_seconds = sleep_seconds
        self.token = None
        self.last_request_at = 0.0

    def refresh_token(self):
        print("Authenticating with ERCOT Public API...")
        self.token = authenticate(self.username, self.password)

    def _throttle(self):
        elapsed = time.monotonic() - self.last_request_at
        wait = self.sleep_seconds - elapsed
        if wait > 0:
            time.sleep(wait)

    def get(self, params: dict, retries: int = 5, *, endpoint=ENDPOINT, binary=False):
        # Archive URLs originate in ERCOT metadata. Never send credentials to
        # an arbitrary host or an unexpected API path.
        url = endpoint if endpoint.startswith("https://") else API_BASE + endpoint
        if not url.startswith(API_BASE + "/"):
            raise ValueError("Download URL must stay within ERCOT public-reports API")
        if params:
            url += ("&" if "?" in url else "?") + urllib.parse.urlencode(params)
        if not self.token:
            self.refresh_token()
        for attempt in range(1, retries + 1):
            self._throttle()
            headers = {
                "Authorization": f"Bearer {self.token}",
                "Ocp-Apim-Subscription-Key": self.subscription_key,
                # ERCOT's public-reports gateway negotiates archive downloads
                # with the same JSON media type as metadata requests. The body
                # is still ZIP bytes when ?download=<docId> is present.
                "Accept": "application/json",
                "User-Agent": "University-of-Tsukuba-ADER-Research/1.0",
            }
            try:
                result = request_json(url, headers=headers, timeout=180, binary=binary)
                self.last_request_at = time.monotonic()
                return result
            except RuntimeError as exc:
                msg = str(exc)
                self.last_request_at = time.monotonic()
                if "HTTP 401" in msg and attempt < retries:
                    print("Token rejected/expired; obtaining a new token...")
                    self.refresh_token()
                    continue
                if "HTTP 429" in msg and attempt < retries:
                    backoff = max(10, 5 * attempt)
                    print(f"Rate limited; waiting {backoff}s...")
                    time.sleep(backoff)
                    continue
                if any(code in msg for code in ("HTTP 500", "HTTP 502", "HTTP 503", "HTTP 504")) and attempt < retries:
                    backoff = 5 * attempt
                    print(f"ERCOT server error; retrying in {backoff}s...")
                    time.sleep(backoff)
                    continue
                raise
        raise RuntimeError("Request failed after retries")


def field_names(response: dict) -> list[str]:
    return [str(f.get("name", "")) for f in response.get("fields", [])]


def normalize_core_columns(fields: list[str]) -> list[str]:
    lookup = {f.lower(): f for f in fields}
    selected = []
    for wanted in CORE_COLUMNS:
        actual = lookup.get(wanted.lower())
        if actual:
            selected.append(actual)
    return selected


def rows_as_dicts(response: dict):
    fields = field_names(response)
    data = response.get("data", [])
    for row in data:
        if isinstance(row, dict):
            yield row
        else:
            # ERCOT row-level API returns a list of lists, with schema in "fields".
            yield dict(zip(fields, row))


def safe_float(value):
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def load_resources(path: str | None, cli_resources: list[str] | None) -> list[str]:
    if cli_resources:
        return list(dict.fromkeys(r.strip() for r in cli_resources if r.strip()))
    if path:
        p = Path(path)
        if p.exists():
            resources = []
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                resources.append(line)
            if resources:
                return list(dict.fromkeys(resources))
    return DEFAULT_RESOURCES.copy()


def estimate(resources: list[str], start: date, end: date, chunk_days: int):
    days = (end - start).days + 1
    expected_rows = len(resources) * days * 288
    chunks_per_resource = math.ceil(days / chunk_days)
    minimum_requests = len(resources) * chunks_per_resource
    return days, expected_rows, minimum_requests


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Download only ERCOT NP3-965-ER Load Resource SCED rows for specified ADER resources. "
            "No full daily disclosure ZIP downloads."
        )
    )
    parser.add_argument("--username", default=os.getenv("ERCOT_USERNAME"), help="ERCOT API Explorer email. Can also set ERCOT_USERNAME.")
    parser.add_argument("--subscription-key", default=os.getenv("ERCOT_SUBSCRIPTION_KEY"), help="ERCOT Public API primary subscription key. Can also set ERCOT_SUBSCRIPTION_KEY.")
    parser.add_argument("--password", default=os.getenv("ERCOT_PASSWORD"), help=argparse.SUPPRESS)
    parser.add_argument("--resource", action="append", help="ADER Resource Name. Repeat for multiple resources.")
    parser.add_argument("--resources-file", default="resources.txt", help="Text file with one Resource Name per line. Default: resources.txt")
    parser.add_argument("--start", default=API_DATA_START.isoformat(), help="SCED date start YYYY-MM-DD. Row-level API starts 2023-12-11.")
    parser.add_argument("--end", default=None, help="SCED date end YYYY-MM-DD. Default: today minus 60 days.")
    parser.add_argument("--chunk-days", type=int, default=DEFAULT_CHUNK_DAYS, help="Days per API query. Default: 31")
    parser.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE, help="Rows per API page. Default: 10000")
    parser.add_argument("--sleep", type=float, default=DEFAULT_SLEEP_SECONDS, help="Minimum seconds between API requests. Default: 2.1")
    parser.add_argument("--out-dir", default="ercot_ader_sced_output", help="Output directory")
    parser.add_argument("--all-columns", action="store_true", help="Also save a second CSV containing every API column.")
    parser.add_argument("--dry-run", action="store_true", help="Show the request plan without authenticating or downloading.")
    parser.add_argument("--test-days", type=int, default=0, help="For a quick test, only download N days starting at --start.")
    args = parser.parse_args()

    if args.chunk_days < 1:
        parser.error("--chunk-days must be >= 1")
    if args.page_size < 1:
        parser.error("--page-size must be >= 1")

    resources = load_resources(args.resources_file, args.resource)
    start = parse_date(args.start)
    latest_default = date.today() - timedelta(days=60)
    end = parse_date(args.end) if args.end else latest_default

    if args.test_days:
        end = min(end, start + timedelta(days=args.test_days - 1))

    if start < API_DATA_START:
        eprint(
            f"WARNING: row-level Public API data for NP3-965-ER starts {API_DATA_START}. "
            f"Clamping requested start {start} to {API_DATA_START}."
        )
        start = API_DATA_START
    if end > latest_default:
        eprint(
            f"NOTE: NP3-965-ER is a 60-day disclosure product. Requested end {end} may not be available; "
            f"the current conservative latest date is {latest_default}."
        )
    if end < start:
        parser.error(f"End date {end} is before start date {start}")

    days, expected_rows, min_requests = estimate(resources, start, end, args.chunk_days)
    print("ERCOT ADER SCED direct-query plan")
    print(f"  Endpoint : {API_BASE}{ENDPOINT}")
    print(f"  Resources: {', '.join(resources)}")
    print(f"  SCED dates: {start} through {end} ({days} days)")
    print(f"  Approx upper-bound rows at 5-min cadence: {expected_rows:,}")
    print(f"  Minimum API requests before pagination: {min_requests:,}")
    print("  Full daily NP3-965-ER ZIPs: NOT downloaded")

    if args.dry_run:
        return 0

    username = args.username or input("ERCOT API Explorer email: ").strip()
    password = args.password or getpass.getpass("ERCOT API password (not stored): ")
    subscription_key = args.subscription_key or getpass.getpass("ERCOT Public API subscription key (not stored): ").strip()
    if not username or not password or not subscription_key:
        raise SystemExit("Username, password, and subscription key are required.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    core_path = out_dir / "ader_sced_dispatch.csv"
    all_path = out_dir / "ader_sced_all_columns.csv"
    meta_path = out_dir / "metadata.json"
    log_path = out_dir / "run.log"

    client = ErcotClient(username, password, subscription_key, args.sleep)

    core_file = core_path.open("w", newline="", encoding="utf-8-sig")
    all_file = all_path.open("w", newline="", encoding="utf-8-sig") if args.all_columns else None
    core_writer = None
    all_writer = None
    all_fields = None
    core_fields = None
    total_rows = 0
    rows_by_resource = {r: 0 for r in resources}
    query_log = []

    try:
        for resource_index, resource in enumerate(resources, 1):
            print(f"\n[{resource_index}/{len(resources)}] Resource {resource}")
            for chunk_start, chunk_end in chunks(start, end, args.chunk_days):
                page = 1
                params = {
                    "resourceName": resource,
                    "SCEDTimestampFrom": iso_start(chunk_start),
                    "SCEDTimestampTo": iso_end_exclusive(chunk_end),
                    "size": args.page_size,
                    "page": page,
                    "sort": "SCEDTimestamp",
                    "dir": "ASC",
                }
                print(f"  {chunk_start} .. {chunk_end}", end="", flush=True)
                response = client.get(params)

                if "_meta" not in response or "data" not in response:
                    raise RuntimeError(f"Unexpected ERCOT response keys: {list(response.keys())}")

                current_fields = field_names(response)
                if all_fields is None:
                    all_fields = current_fields
                    core_fields = normalize_core_columns(all_fields)
                    if "basePoint" not in [f for f in core_fields]:
                        # Case-insensitive check for a schema change.
                        if "basepoint" not in {f.lower() for f in core_fields}:
                            raise RuntimeError(
                                "ERCOT response does not contain basePoint. Fields received: "
                                + ", ".join(all_fields)
                            )
                    core_writer = csv.DictWriter(core_file, fieldnames=core_fields, extrasaction="ignore")
                    core_writer.writeheader()
                    if all_file:
                        all_writer = csv.DictWriter(all_file, fieldnames=all_fields, extrasaction="ignore")
                        all_writer.writeheader()
                elif current_fields != all_fields:
                    raise RuntimeError("ERCOT API schema changed during this run; stopping to avoid misaligned CSV output.")

                meta = response.get("_meta", {})
                total_pages = int(meta.get("totalPages") or 1)
                chunk_rows = 0

                while True:
                    for row in rows_as_dicts(response):
                        # Defensive resource check even though resourceName is a server-side filter.
                        row_resource = str(row.get("resourceName", "")).strip()
                        if row_resource and row_resource.upper() != resource.upper():
                            continue
                        core_writer.writerow({k: row.get(k, "") for k in core_fields})
                        if all_writer:
                            all_writer.writerow({k: row.get(k, "") for k in all_fields})
                        chunk_rows += 1
                        total_rows += 1
                        rows_by_resource[resource] += 1

                    if page >= total_pages:
                        break
                    page += 1
                    params["page"] = page
                    response = client.get(params)

                print(f"  -> {chunk_rows:,} rows ({total_pages} page{'s' if total_pages != 1 else ''})")
                query_log.append(
                    {
                        "resourceName": resource,
                        "SCEDTimestampFrom": iso_start(chunk_start),
                        "SCEDTimestampTo": iso_end_exclusive(chunk_end),
                        "rows": chunk_rows,
                        "pages": total_pages,
                    }
                )
                core_file.flush()
                if all_file:
                    all_file.flush()
    finally:
        core_file.close()
        if all_file:
            all_file.close()

    # Drop exact duplicate lines while keeping the header. This is conservative and does
    # not collapse records merely because timestamp/resource match.
    def dedupe_csv_exact(path: Path):
        tmp = path.with_suffix(path.suffix + ".tmp")
        with path.open("r", newline="", encoding="utf-8-sig") as src, tmp.open("w", newline="", encoding="utf-8-sig") as dst:
            reader = csv.reader(src)
            writer = csv.writer(dst)
            try:
                header = next(reader)
            except StopIteration:
                return 0
            writer.writerow(header)
            seen = set()
            kept = 0
            for row in reader:
                key = tuple(row)
                if key in seen:
                    continue
                seen.add(key)
                writer.writerow(row)
                kept += 1
        tmp.replace(path)
        return kept

    core_kept = dedupe_csv_exact(core_path)
    all_kept = dedupe_csv_exact(all_path) if args.all_columns else None

    metadata = {
        "createdAt": datetime.now().isoformat(timespec="seconds"),
        "source": "ERCOT Public API",
        "emilProduct": "NP3-965-ER",
        "artifact": "60 Day Load Resource Data in SCED",
        "endpoint": API_BASE + ENDPOINT,
        "resources": resources,
        "scedDateStart": start.isoformat(),
        "scedDateEnd": end.isoformat(),
        "apiRowLevelStart": API_DATA_START.isoformat(),
        "notePreApi": (
            "SCED dates before 2023-12-11 are not available through row-level Public API query parameters; "
            "use historic archive files for that earlier period."
        ),
        "rowsWrittenBeforeExactDedupe": total_rows,
        "rowsInDispatchCsvAfterExactDedupe": core_kept,
        "rowsByResourceBeforeExactDedupe": rows_by_resource,
        "coreColumns": core_fields,
        "allColumnsSaved": bool(args.all_columns),
        "allColumnsRowCountAfterExactDedupe": all_kept,
        "queries": query_log,
        "sources": [
            "https://developer.ercot.com/applications/pubapi/user-guide/registration-and-authentication/",
            "https://developer.ercot.com/applications/pubapi/known-limits/",
            "https://www.ercot.com/mp/data-products/data-product-details?id=NP3-965-ER",
        ],
    }
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    log_path.write_text(
        "Download completed successfully.\n"
        f"Rows written: {core_kept}\n"
        + "\n".join(f"{k}: {v}" for k, v in rows_by_resource.items())
        + "\n",
        encoding="utf-8",
    )

    zip_path = out_dir.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for p in [core_path, meta_path, log_path]:
            zf.write(p, arcname=p.name)
        if args.all_columns:
            zf.write(all_path, arcname=all_path.name)

    print("\nDone.")
    print(f"  Dispatch CSV: {core_path}")
    if args.all_columns:
        print(f"  All columns : {all_path}")
    print(f"  Final ZIP   : {zip_path}")
    print(f"  Rows        : {core_kept:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
