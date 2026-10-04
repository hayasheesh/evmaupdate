"""Fetch one operating day of ERCOT disclosures to see where an ESR's own plan is.

The AEMO command library is a unit's dispatch target minus its own plan. For
ERCOT the dispatch target is the SCED Base Point; this probe looks for the
plan. For each product it downloads the posting that carries the requested
operating day (posted 60 days later), writes every CSV member's columns and
first rows to ``columns.txt``, and keeps ESR and COP members whole.

A product that cannot be listed or downloaded is recorded and skipped. No
waveform, bank or training is built.
"""
from __future__ import annotations

import argparse
import csv
from datetime import date, timedelta
import getpass
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

PRODUCTS = {
    "np3-966-er": "60-Day DAM Disclosure",
    "np3-991-ex": "60-Day COP All Updates",
    "np1-301": "60-Day COP Adjustment Period Snapshot",
    "np3-965-er": "60-Day SCED Disclosure",
}
PREVIEW_ROWS = 3


def keep_whole(member: str) -> bool:
    key = re.sub(r"[^a-z0-9]", "", Path(member).name.lower())
    return "esr" in key or "cop" in key


def list_postings(client, product: str, operating_day: date) -> list[dict]:
    posted = operating_day + timedelta(days=60)
    params = {
        "postDatetimeFrom": f"{posted}T00:00:00",
        "postDatetimeTo": f"{posted + timedelta(days=2)}T00:00:00",
        "size": 1000,
        "page": 1,
    }
    response = client.get(params, endpoint=f"/archive/{product}")
    if "archives" not in response:
        raise ValueError(f"unexpected archive-list schema for {product}")
    return sorted(response["archives"], key=lambda item: item["postDatetime"])


def inspect_zip(blob: bytes, out_dir: Path, columns_log) -> list[dict]:
    saved = []
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        for member in archive.namelist():
            if member.lower().endswith(".zip"):
                saved += inspect_zip(archive.read(member), out_dir, columns_log)
                continue
            if not member.lower().endswith(".csv"):
                columns_log.write(f"\n## {member} (not CSV)\n")
                continue
            with archive.open(member) as source:
                text = io.TextIOWrapper(source, encoding="utf-8-sig")
                reader = csv.reader(text)
                header = next(reader, [])
                preview = [row for _, row in zip(range(PREVIEW_ROWS), reader)]
            columns_log.write(f"\n## {member}\n")
            columns_log.write(f"columns ({len(header)}): {header}\n")
            for row in preview:
                columns_log.write(f"  {row}\n")
            record = {"member": member, "columns": header}
            if keep_whole(member):
                target = out_dir / Path(member).name
                target.write_bytes(archive.read(member))
                record["saved"] = target.name
            saved.append(record)
    return saved


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operating-date", type=date.fromisoformat, default=date(2026, 3, 10))
    parser.add_argument("--products", default=",".join(PRODUCTS))
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path(__file__).resolve().parent / "ercot_plan_probe_output",
    )
    parser.add_argument("--sleep", type=float, default=DEFAULT_SLEEP_SECONDS)
    args = parser.parse_args(argv)
    products = [p.strip().lower() for p in args.products.split(",") if p.strip()]
    print(f"Operating date {args.operating_date}; products {products}")
    print("Keeps ESR/COP members whole and the columns of every member. No bank is built.")

    username = os.getenv("ERCOT_USERNAME") or input("ERCOT API Explorer email: ").strip()
    password = os.getenv("ERCOT_PASSWORD") or getpass.getpass("ERCOT API password (not stored): ")
    key = os.getenv("ERCOT_SUBSCRIPTION_KEY") or getpass.getpass("ERCOT subscription key (not stored): ").strip()
    if not username or not password or not key:
        parser.error("Email, password and subscription key are required")
    client = ErcotClient(username, password, key, args.sleep)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"operating_date": str(args.operating_date), "products": {}}
    for product in products:
        product_dir = args.out_dir / product
        product_dir.mkdir(parents=True, exist_ok=True)
        entry = {"name": PRODUCTS.get(product, ""), "postings": []}
        summary["products"][product] = entry
        try:
            postings = list_postings(client, product, args.operating_date)
        except Exception as exc:  # recorded, not fatal: the product id may differ
            entry["error"] = f"listing failed: {exc}"
            print(f"[{product}] listing failed: {exc}")
            continue
        if not postings:
            entry["error"] = "no posting in the 60-62 day window"
            print(f"[{product}] no posting found")
            continue
        posting = postings[0]
        url = posting["_links"]["endpoint"]["href"]
        print(f"[{product}] downloading posting {posting['postDatetime']}", flush=True)
        try:
            blob = client.get({}, endpoint=url, binary=True)
            with (product_dir / "columns.txt").open("w", encoding="utf-8") as log:
                log.write(f"# {product} {PRODUCTS.get(product, '')} posting {posting['postDatetime']}\n")
                members = inspect_zip(blob, product_dir, log)
        except Exception as exc:
            entry["error"] = f"download failed: {exc}"
            print(f"[{product}] download failed: {exc}")
            continue
        entry["postings"].append({"postDatetime": posting["postDatetime"], "url": url, "members": members})
        print(f"[{product}] {len(members)} CSV member(s); kept whole: "
              f"{[m['saved'] for m in members if 'saved' in m]}")
        (args.out_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Done: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
