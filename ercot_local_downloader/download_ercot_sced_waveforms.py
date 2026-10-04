"""Download NP3-965-ER archives and retain CLR/ESR waveform inputs.

Reuses the existing API authentication. Does not assume an undocumented ESR
row endpoint. No request, authentication or output creation with --dry-run.
Default scope: all load resources and all ESRs from RTC+B onward. Generation
data is opt-in. Does NOT build synthetic training scenarios or a bid bank.
"""
from __future__ import annotations

import argparse
import csv
from datetime import date, datetime, timedelta
import getpass
import hashlib
import io
import json
import os
from pathlib import Path
import re
import zipfile

try:
    from .download_ercot_ader_sced_api import ErcotClient, DEFAULT_SLEEP_SECONDS
except ImportError:
    from download_ercot_ader_sced_api import ErcotClient, DEFAULT_SLEEP_SECONDS

RTC_START = date(2025, 12, 5)
ARCHIVE_ENDPOINT = "/archive/np3-965-er"
FIELDS = ["SCEDTimestamp", "repeatHourFlag", "resourceName", "resource_kind",
          "telResStatus", "basePoint", "maxPowerConsumption", "lowPowerConsumption",
          "HSL", "LSL", "source_posting_date", "source_document_url", "source_zip_member"]
ALIASES = {
    "SCEDTimestamp": ("scedtimestamp",), "repeatHourFlag": ("repeatedhourflag", "repeathourflag"),
    "resourceName": ("resourcename",), "telResStatus": ("telemeteredresourcestatus", "telresstatus"),
    "basePoint": ("basepoint",), "HSL": ("hsl", "highsustainedlimit"),
    "LSL": ("lsl", "lowsustainedlimit"),
    "maxPowerConsumption": ("maxpowerconsumption", "maximumpowerconsumption", "mpc"),
    "lowPowerConsumption": ("lowpowerconsumption", "lpc"),
}


def member_kind(name: str) -> str | None:
    key = re.sub(r"[^a-z0-9]", "", Path(name).name.lower())
    if not name.lower().endswith(".csv"):
        return None
    for marker, kind in [("loadresourcedatainsced", "CLR"),
                         ("esrdatainsced", "ESR"),
                         ("generationresourcedatainsced", "GEN"),
                         ("genresourcedatainsced", "GEN")]:
        if marker in key:
            return kind
    return None


def normalize_timestamp(value: str) -> datetime:
    value = value.strip()
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        for fmt in ("%m/%d/%Y %H:%M:%S", "%m/%d/%Y %H:%M:%S.%f"):
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                pass
    raise ValueError(f"Invalid SCED timestamp: {value!r}")


def extract_document(blob: bytes, *, output_dir: Path, document_url: str,
                     posting_date: str, start: date, end: date,
                     include_generation: bool = False) -> dict:
    """Stream just the relevant CSV members; retain no huge downloaded ZIP."""
    if not zipfile.is_zipfile(io.BytesIO(blob)):
        raise RuntimeError(
            "ERCOT archive request returned a non-ZIP response; "
            "check the API response and Accept header."
        )
    document_id = hashlib.sha256(document_url.encode()).hexdigest()[:24]
    written = []
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        for member in archive.namelist():
            kind = member_kind(member)
            if kind is None or (kind == "GEN" and not include_generation):
                continue
            # Include member identity: supplemental archives can contain many
            # operating days of the same kind in one document.
            member_id = hashlib.sha256(member.encode()).hexdigest()[:12]
            filename = f"{document_id}_{kind}_{member_id}.csv"
            destination = output_dir / filename
            partial = destination.with_suffix(".csv.part")
            count = 0
            with archive.open(member) as source, io.TextIOWrapper(source, encoding="utf-8-sig") as text:
                reader = csv.DictReader(text)
                headers = {re.sub(r"[^a-z0-9]", "", c.lower()): c for c in (reader.fieldnames or [])}
                columns = {key: next((headers[a] for a in aliases if a in headers), None)
                           for key, aliases in ALIASES.items()}
                needed = ["SCEDTimestamp", "resourceName", "telResStatus", "basePoint"] + (
                    ["maxPowerConsumption", "lowPowerConsumption"] if kind == "CLR" else ["HSL", "LSL"])
                if missing := [c for c in needed if not columns[c]]:
                    raise ValueError(f"{member}: missing columns {missing}")
                with partial.open("w", newline="", encoding="utf-8") as target:
                    writer = csv.DictWriter(target, fieldnames=FIELDS)
                    writer.writeheader()
                    for raw in reader:
                        timestamp = normalize_timestamp(raw[columns["SCEDTimestamp"]])
                        if not start <= timestamp.date() <= end:
                            continue
                        row = {key: raw.get(column, "") if column else "" for key, column in columns.items()}
                        row.update(SCEDTimestamp=timestamp.isoformat(), resource_kind=kind,
                                   source_posting_date=posting_date, source_document_url=document_url,
                                   source_zip_member=member)
                        if not str(row["resourceName"]).strip():
                            raise ValueError(f"{member}: missing resource name")
                        writer.writerow(row)
                        count += 1
            if destination.exists():
                # Resume a crash after CSV commit but before manifest commit.
                # Only an identical reconstruction may reuse that file.
                if file_hash(partial) != file_hash(destination):
                    raise FileExistsError(f"Existing output differs: {destination}")
                partial.unlink()
            else:
                partial.replace(destination)
            written.append({"file": filename, "kind": kind, "rows": count,
                            "sha256": file_hash(destination)})
    return {"url": document_url, "postDatetime": posting_date, "files": written}


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_manifest(path: Path, state: dict) -> None:
    partial = path.with_suffix(".json.part")
    partial.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    partial.replace(path)


def list_archives(client, start: date, *, through: date) -> list[dict]:
    """Posting dates, not operating dates; include later corrective postings."""
    params = {"postDatetimeFrom": f"{start + timedelta(days=60)}T00:00:00",
              "postDatetimeTo": f"{through + timedelta(days=1)}T00:00:00", "size": 1000, "page": 1}
    archives = []
    while True:
        response = client.get(params, endpoint=ARCHIVE_ENDPOINT)
        if "archives" not in response or "_meta" not in response:
            raise ValueError("Unexpected archive-list schema; no rows downloaded")
        archives.extend(response["archives"])
        if params["page"] >= int(response["_meta"].get("totalPages") or 1):
            break
        params["page"] += 1
    return sorted(archives, key=lambda item: item["postDatetime"])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, default=RTC_START)
    parser.add_argument("--end", type=date.fromisoformat)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "ercot_sced_rtc_output")
    parser.add_argument("--include-generation", action="store_true")
    parser.add_argument("--test-days", type=int, default=0)
    parser.add_argument("--sleep", type=float, default=DEFAULT_SLEEP_SECONDS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    state_path = args.out_dir / "manifest.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    end = args.end or (date.fromisoformat(state["end"]) if state else date.today() - timedelta(days=60))
    if args.test_days:
        end = min(end, args.start + timedelta(days=args.test_days - 1))
    if args.start < RTC_START or end < args.start or args.sleep < 2.1 or args.test_days < 0:
        parser.error("Require start>=2025-12-05, end>=start, sleep>=2.1, test-days>=0")
    config = {"start": str(args.start), "end": str(end), "include_generation": args.include_generation}
    if state and any(state.get(k) != v for k, v in config.items()):
        parser.error("Existing download has a different range/scope; use a new --out-dir")
    print(f"Operating dates: {args.start} .. {end}; all CLR + ESR" + (" + GEN" if args.include_generation else ""))
    print("API archive route: NP3-965-ER daily/supplemental ZIPs; retain selected CSV columns only.")
    print("Original rows retained. No 5-minute resampling, bank creation or training in this command.")
    if args.dry_run:
        print("Dry run: no authentication, requests or writes.")
        return 0
    username = os.getenv("ERCOT_USERNAME") or input("ERCOT API Explorer email: ").strip()
    password = os.getenv("ERCOT_PASSWORD") or getpass.getpass("ERCOT API password (not stored): ")
    key = os.getenv("ERCOT_SUBSCRIPTION_KEY") or getpass.getpass("ERCOT subscription key (not stored): ").strip()
    if not username or not password or not key:
        parser.error("Email, password and subscription key are required")
    client = ErcotClient(username, password, key, args.sleep)
    archives = list_archives(client, args.start, through=date.today())
    documents = args.out_dir / "documents"
    documents.mkdir(parents=True, exist_ok=True)
    state = state or {**config, "documents": {}}
    state["download_complete"] = False
    save_manifest(state_path, state)
    for record in archives:
        url = record["_links"]["endpoint"]["href"]
        if url in state["documents"]:
            for saved in state["documents"][url]["files"]:
                path = documents / saved["file"]
                if not path.exists() or file_hash(path) != saved["sha256"]:
                    raise ValueError(f"Resume integrity check failed: {path}")
            continue
        print(f"Downloading posting {record['postDatetime']}", flush=True)
        blob = client.get({}, endpoint=url, binary=True)
        result = extract_document(blob, output_dir=documents, document_url=url,
                                  posting_date=record["postDatetime"], start=args.start, end=end,
                                  include_generation=args.include_generation)
        state["documents"][url] = result
        save_manifest(state_path, state)
        print(f"  retained rows: {sum(f['rows'] for f in result['files']):,}", flush=True)
    counts = {kind: sum(f["rows"] for d in state["documents"].values() for f in d["files"] if f["kind"] == kind)
              for kind in ("CLR", "ESR", "GEN")}
    print(f"Completed: {counts}; {args.out_dir}")
    if not counts["CLR"] or not counts["ESR"]:
        raise RuntimeError("Required CLR/ESR rows missing; do not treat this download as complete")
    state["download_complete"] = True
    save_manifest(state_path, state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
