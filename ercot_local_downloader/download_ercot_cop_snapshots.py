"""Download NP1-301 60-Day COP Adjustment Period Snapshots from RTC+B onward.

Each snapshot holds, for every resource and operating hour, the Current
Operating Plan as it stood one hour before that hour. For an ESR its
``Hour Beginning Planned SOC`` is the QSE's own energy plan; the hour-to-hour
difference is its planned output, which the ERCOT command library subtracts
from the SCED Base Point.

Files are kept whole. Reuses the existing API authentication; credentials are
entered by the operator and never stored. Resumable: a rerun skips postings
whose files are already saved with a matching hash. No bank is built.
"""
from __future__ import annotations

import argparse
import csv
from datetime import date, timedelta
import getpass
import hashlib
import io
import json
import os
from pathlib import Path
import zipfile

try:
    from .download_ercot_ader_sced_api import ErcotClient, DEFAULT_SLEEP_SECONDS
except ImportError:
    from download_ercot_ader_sced_api import ErcotClient, DEFAULT_SLEEP_SECONDS

RTC_START = date(2025, 12, 5)
ARCHIVE_ENDPOINT = "/archive/np1-301"
REQUIRED_COLUMNS = ("Delivery Date", "Resource Name", "Hour Ending", "Hour Beginning Planned SOC")


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


def list_postings(client, start: date, through: date) -> list[dict]:
    params = {
        "postDatetimeFrom": f"{start + timedelta(days=60)}T00:00:00",
        "postDatetimeTo": f"{through + timedelta(days=1)}T00:00:00",
        "size": 1000,
        "page": 1,
    }
    postings = []
    while True:
        response = client.get(params, endpoint=ARCHIVE_ENDPOINT)
        if "archives" not in response or "_meta" not in response:
            raise ValueError("Unexpected archive-list schema; nothing downloaded")
        postings.extend(response["archives"])
        if params["page"] >= int(response["_meta"].get("totalPages") or 1):
            break
        params["page"] += 1
    return sorted(postings, key=lambda item: item["postDatetime"])


def save_members(blob: bytes, documents: Path, posting: str) -> list[dict]:
    if not zipfile.is_zipfile(io.BytesIO(blob)):
        raise RuntimeError("ERCOT archive request returned a non-ZIP response")
    saved = []
    stamp = posting.replace(":", "").replace("-", "")[:15]
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        for member in archive.namelist():
            if not member.lower().endswith(".csv"):
                continue
            data = archive.read(member)
            reader = csv.reader(io.StringIO(data.decode("utf-8-sig")))
            header = next(reader, [])
            missing = [c for c in REQUIRED_COLUMNS if c not in header]
            if missing:
                raise ValueError(f"{member}: missing columns {missing}")
            date_index = header.index("Delivery Date")
            delivery_dates = sorted({row[date_index] for row in reader if len(row) > date_index})
            target = documents / f"{stamp}_{Path(member).name}"
            partial = target.with_suffix(".csv.part")
            partial.write_bytes(data)
            if target.exists() and file_hash(target) != file_hash(partial):
                raise FileExistsError(f"Existing output differs: {target}")
            partial.replace(target)
            saved.append({
                "file": target.name,
                "member": member,
                "delivery_dates": delivery_dates,
                "sha256": file_hash(target),
            })
    return saved


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, default=RTC_START)
    parser.add_argument("--end", type=date.fromisoformat)
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path(__file__).resolve().parent / "ercot_cop_snapshot_output",
    )
    parser.add_argument("--sleep", type=float, default=DEFAULT_SLEEP_SECONDS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    state_path = args.out_dir / "manifest.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    end = args.end or (date.fromisoformat(state["end"]) if state else date.today() - timedelta(days=60))
    if args.start < RTC_START or end < args.start or args.sleep < 2.1:
        parser.error("Require start>=2025-12-05, end>=start, sleep>=2.1")
    config = {"start": str(args.start), "end": str(end)}
    if state and any(state.get(k) != v for k, v in config.items()):
        parser.error("Existing download has a different range; use a new --out-dir")
    print(f"Operating dates: {args.start} .. {end}; NP1-301 COP Adjustment Period Snapshot, files kept whole")
    if args.dry_run:
        print("Dry run: no authentication, requests or writes.")
        return 0
    username = os.getenv("ERCOT_USERNAME") or input("ERCOT API Explorer email: ").strip()
    password = os.getenv("ERCOT_PASSWORD") or getpass.getpass("ERCOT API password (not stored): ")
    key = os.getenv("ERCOT_SUBSCRIPTION_KEY") or getpass.getpass("ERCOT subscription key (not stored): ").strip()
    if not username or not password or not key:
        parser.error("Email, password and subscription key are required")
    client = ErcotClient(username, password, key, args.sleep)
    postings = list_postings(client, args.start, date.today())
    documents = args.out_dir / "documents"
    documents.mkdir(parents=True, exist_ok=True)
    state = state or {**config, "documents": {}}
    state["download_complete"] = False
    save_manifest(state_path, state)
    for posting in postings:
        url = posting["_links"]["endpoint"]["href"]
        if url in state["documents"]:
            for saved in state["documents"][url]["files"]:
                path = documents / saved["file"]
                if not path.exists() or file_hash(path) != saved["sha256"]:
                    raise ValueError(f"Resume integrity check failed: {path}")
            continue
        print(f"Downloading posting {posting['postDatetime']}", flush=True)
        blob = client.get({}, endpoint=url, binary=True)
        files = save_members(blob, documents, posting["postDatetime"])
        state["documents"][url] = {"postDatetime": posting["postDatetime"], "files": files}
        save_manifest(state_path, state)
        print(f"  delivery dates: {sorted({d for f in files for d in f['delivery_dates']})}", flush=True)
    covered = sorted({d for doc in state["documents"].values() for f in doc["files"] for d in f["delivery_dates"]})
    print(f"Completed: {len(covered)} delivery date(s); {args.out_dir}")
    state["download_complete"] = True
    state["delivery_dates"] = covered
    save_manifest(state_path, state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
