#!/usr/bin/env python3
import argparse
import csv
import io
import json
import re
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

import requests

DOC_LIST_URL = "https://www.ercot.com/misapp/servlets/IceDocListJsonWS"
DOWNLOAD_URL = "https://www.ercot.com/misdownload/servlets/mirDownload?doclookupId={}"
REPORT_TYPE_ID = 13052
DEFAULT_RESOURCES = ["AR_ALD1", "BOERNE_ALD1"]

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126 Safari/537.36"
HEADERS = {
    "User-Agent": UA,
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "X-Requested-With": "XMLHttpRequest",
    "Referer": "https://www.ercot.com/mp/data-products/data-product-details?id=NP3-965-ER",
}


def parse_dt(s):
    if not s:
        return None
    # Handle many variants seen in ERCOT metadata.
    for fmt in (
        "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%m/%d/%Y %H:%M:%S",
        "%Y-%m-%d", "%m/%d/%Y", "%b %d, %Y", "%d-%b-%Y"
    ):
        try:
            return datetime.strptime(str(s)[:19], fmt)
        except Exception:
            pass
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        return None


def get_doc_list(session):
    r = session.get(DOC_LIST_URL, params={"reportTypeId": REPORT_TYPE_ID, "_": int(time.time())}, headers=HEADERS, timeout=60)
    r.raise_for_status()
    data = r.json()
    docs = data.get("ListDocsByRptTypeRes", {}).get("DocumentList", [])
    out = []
    for item in docs:
        d = item.get("Document", item)
        doc_id = d.get("DocID") or d.get("docID") or d.get("docId")
        if doc_id is None:
            continue
        out.append(d)
    return out


def doc_date(d):
    candidates = [
        d.get("PublishDate"), d.get("PostedDate"), d.get("Created"), d.get("created"),
        d.get("ReportDate"), d.get("OperatingDate"), d.get("Date")
    ]
    for x in candidates:
        dt = parse_dt(x)
        if dt:
            return dt
    # Fall back to dates embedded in names, if present.
    txt = " ".join(str(d.get(k, "")) for k in ("FriendlyName", "ConstructedName", "FileName"))
    m = re.search(r"(\d{4})[-_/](\d{2})[-_/](\d{2})", txt)
    if m:
        return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    return None


def find_col(header, names):
    norm = {re.sub(r"[^a-z0-9]", "", h.lower()): i for i, h in enumerate(header)}
    for name in names:
        key = re.sub(r"[^a-z0-9]", "", name.lower())
        if key in norm:
            return norm[key]
    return None


def process_csv_bytes(raw, exact_resources, discover_regex, known_writer, candidate_writer, source_meta):
    # ERCOT files are normally ASCII/UTF-8; tolerate BOM/latin1 fallbacks.
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("latin1")
    reader = csv.reader(io.StringIO(text))
    try:
        header = next(reader)
    except StopIteration:
        return 0, 0, None

    res_idx = find_col(header, ["Resource Name", "ResourceName", "RESOURCE_NAME", "Resource"])
    if res_idx is None:
        return 0, 0, header

    known_count = 0
    cand_count = 0
    for row in reader:
        if res_idx >= len(row):
            continue
        rname = row[res_idx].strip()
        if rname in exact_resources:
            known_writer.writerow(row + source_meta)
            known_count += 1
        elif discover_regex and discover_regex.search(rname):
            candidate_writer.writerow(row + source_meta)
            cand_count += 1
    return known_count, cand_count, header


def main():
    ap = argparse.ArgumentParser(description="Download ERCOT NP3-965-ER daily ZIPs from a whitelisted connection, extract Load Resource SCED records, and bundle ADER rows.")
    ap.add_argument("--out", default="ercot_ader_sced_output", help="Output directory")
    ap.add_argument("--from-date", default="2023-10-23", help="Posting date lower bound YYYY-MM-DD. 2023-10-23 roughly covers ADER market start + 60-day lag.")
    ap.add_argument("--to-date", default=None, help="Posting date upper bound YYYY-MM-DD")
    ap.add_argument("--resource", action="append", help="Exact resource name; may be repeated. Defaults to AR_ALD1 and BOERNE_ALD1.")
    ap.add_argument("--discover-regex", default=r"(?:^|_)ALD\d+$", help="Regex for additional candidate Aggregate Load Resource names. Empty string disables discovery.")
    ap.add_argument("--max-docs", type=int, default=0, help="Limit number of documents for testing; 0 = no limit")
    ap.add_argument("--sleep", type=float, default=0.4, help="Seconds between downloads")
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    state_path = outdir / "progress.json"
    known_csv = outdir / "known_ader_sced.csv"
    cand_csv = outdir / "additional_ald_candidates.csv"
    manifest_csv = outdir / "manifest.csv"
    log_path = outdir / "run.log"

    exact_resources = set(args.resource or DEFAULT_RESOURCES)
    discover_regex = re.compile(args.discover_regex, re.I) if args.discover_regex else None
    from_dt = parse_dt(args.from_date)
    to_dt = parse_dt(args.to_date) if args.to_date else None

    processed = set()
    if state_path.exists():
        try:
            processed = set(json.loads(state_path.read_text(encoding="utf-8")).get("processed_doc_ids", []))
        except Exception:
            pass

    s = requests.Session()
    print("Fetching ERCOT document list...")
    docs = get_doc_list(s)
    print(f"Document list returned {len(docs)} entries.")

    selected = []
    for d in docs:
        did = str(d.get("DocID") or d.get("docID") or d.get("docId"))
        dt = doc_date(d)
        if did in processed:
            continue
        if from_dt and dt and dt < from_dt:
            continue
        if to_dt and dt and dt > to_dt:
            continue
        selected.append((dt or datetime.min, d))
    selected.sort(key=lambda x: x[0])
    if args.max_docs:
        selected = selected[: args.max_docs]
    print(f"Selected {len(selected)} documents to process.")

    known_f = known_csv.open("a", newline="", encoding="utf-8")
    cand_f = cand_csv.open("a", newline="", encoding="utf-8")
    manifest_exists = manifest_csv.exists() and manifest_csv.stat().st_size > 0
    manifest_f = manifest_csv.open("a", newline="", encoding="utf-8")
    known_writer = csv.writer(known_f)
    cand_writer = csv.writer(cand_f)
    manifest_writer = csv.writer(manifest_f)
    if not manifest_exists:
        manifest_writer.writerow(["doc_id", "posting_date", "friendly_name", "constructed_name", "zip_member", "known_rows", "candidate_rows", "http_status", "bytes"])

    header_written_known = known_csv.exists() and known_csv.stat().st_size > 0
    header_written_cand = cand_csv.exists() and cand_csv.stat().st_size > 0

    try:
        for idx, (dt, d) in enumerate(selected, 1):
            did = str(d.get("DocID") or d.get("docID") or d.get("docId"))
            friendly = str(d.get("FriendlyName", ""))
            constructed = str(d.get("ConstructedName", d.get("FileName", "")))
            print(f"[{idx}/{len(selected)}] downloading doc {did} {friendly or constructed}")
            url = DOWNLOAD_URL.format(did)
            try:
                r = s.get(url, headers={**HEADERS, "Accept": "application/zip,application/octet-stream,*/*"}, timeout=180)
                status = r.status_code
                r.raise_for_status()
                blob = r.content
            except Exception as e:
                with log_path.open("a", encoding="utf-8") as lf:
                    lf.write(f"{datetime.now().isoformat()} ERROR doc {did}: {e}\n")
                print(f"  ERROR: {e}", file=sys.stderr)
                continue

            if blob[:2] != b"PK":
                # Often means an HTML Access Denied page rather than a ZIP.
                msg = blob[:500].decode("utf-8", errors="replace")
                with log_path.open("a", encoding="utf-8") as lf:
                    lf.write(f"{datetime.now().isoformat()} NONZIP doc {did} status={status}: {msg!r}\n")
                print("  Received non-ZIP content. Check that this PC's public IPv4 is the whitelisted one.", file=sys.stderr)
                continue

            k_total = 0
            c_total = 0
            found_member = False
            try:
                with zipfile.ZipFile(io.BytesIO(blob)) as zf:
                    members = [m for m in zf.namelist() if re.search(r"Load_Resource_Data_in_SCED.*\.csv$", m, re.I)]
                    for member in members:
                        found_member = True
                        raw = zf.read(member)
                        source_meta = [did, dt.date().isoformat() if dt != datetime.min else "", friendly, constructed, member]
                        # Need header first to initialize output writers exactly once.
                        try:
                            t = raw.decode("utf-8-sig")
                        except UnicodeDecodeError:
                            t = raw.decode("latin1")
                        rdr = csv.reader(io.StringIO(t))
                        try:
                            hdr = next(rdr)
                        except StopIteration:
                            continue
                        if not header_written_known:
                            known_writer.writerow(hdr + ["source_doc_id", "source_posting_date", "source_friendly_name", "source_constructed_name", "source_zip_member"])
                            header_written_known = True
                        if not header_written_cand:
                            cand_writer.writerow(hdr + ["source_doc_id", "source_posting_date", "source_friendly_name", "source_constructed_name", "source_zip_member"])
                            header_written_cand = True
                        # Re-process from raw now that headers exist.
                        kc, cc, _ = process_csv_bytes(raw, exact_resources, discover_regex, known_writer, cand_writer, source_meta)
                        k_total += kc
                        c_total += cc
                        manifest_writer.writerow([did, dt.date().isoformat() if dt != datetime.min else "", friendly, constructed, member, kc, cc, status, len(blob)])
            except zipfile.BadZipFile as e:
                with log_path.open("a", encoding="utf-8") as lf:
                    lf.write(f"{datetime.now().isoformat()} BADZIP doc {did}: {e}\n")
                continue

            if not found_member:
                manifest_writer.writerow([did, dt.date().isoformat() if dt != datetime.min else "", friendly, constructed, "", 0, 0, status, len(blob)])

            processed.add(did)
            state_path.write_text(json.dumps({"processed_doc_ids": sorted(processed)}, indent=2), encoding="utf-8")
            known_f.flush(); cand_f.flush(); manifest_f.flush()
            print(f"  rows: known ADER={k_total}, extra ALD candidates={c_total}")
            time.sleep(args.sleep)
    finally:
        known_f.close(); cand_f.close(); manifest_f.close()

    # Bundle filtered output, not the huge raw source ZIPs.
    bundle = outdir.with_suffix(".zip")
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for p in [known_csv, cand_csv, manifest_csv, state_path, log_path]:
            if p.exists():
                zf.write(p, arcname=p.name)
    print(f"Done. Bundle: {bundle}")
    print(f"Exact resources: {', '.join(sorted(exact_resources))}")

if __name__ == "__main__":
    main()
