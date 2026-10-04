"""Offline protocol tests: no ERCOT credentials or network access."""
from datetime import date
import csv
import io
import json
from pathlib import Path
import zipfile

import pandas as pd
import pytest

from ercot_local_downloader import download_ercot_sced_waveforms as downloader
from ercot_local_downloader import download_ercot_ader_sced_api as api


def _archive(*, missing_base_point=False) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for kind, member, status, upper, lower in [
            ("CLR", "60d_Load_Resource_Data_in_SCED-03-FEB-26.csv", "ONL", "Max Power Consumption", "Low Power Consumption"),
            ("ESR", "60d_ESR_Data_in_SCED-03-FEB-26.csv", "ON", "HSL", "LSL"),
            ("GEN", "60d_Generation_Resource_Data_in_SCED-03-FEB-26.csv", "ON", "HSL", "LSL"),
        ]:
            stream = io.StringIO()
            writer = csv.writer(stream)
            writer.writerow(["SCED Time Stamp", "Repeated Hour Flag", "Resource Name", "Telemetered Resource Status",
                             "INVALID" if missing_base_point else "Base Point", upper, lower])
            writer.writerow(["12/05/2025 00:00:03", "N", kind + "_TEST", status, -2, 5, -5])
            writer.writerow(["12/06/2025 00:00:03", "N", kind + "_TEST", status, 3, 5, -5])
            archive.writestr(member, stream.getvalue())
        archive.writestr("60d_SCED_Energy_Offer_Curve.csv", "not a waveform input")
    return buffer.getvalue()


def test_extracts_all_clr_and_esr_not_generation_or_other_members(tmp_path: Path) -> None:
    result = downloader.extract_document(_archive(), output_dir=tmp_path,
        document_url=api.API_BASE + "/archive/document/1", posting_date="2026-02-03T12:00:00",
        start=date(2025, 12, 5), end=date(2025, 12, 5))
    assert {f["kind"] for f in result["files"]} == {"CLR", "ESR"}
    assert sum(f["rows"] for f in result["files"]) == 2
    for item in result["files"]:
        row = pd.read_csv(tmp_path / item["file"]).iloc[0]
        assert row.SCEDTimestamp == "2025-12-05T00:00:03"
        assert row.basePoint == -2
        assert row.source_posting_date == "2026-02-03T12:00:00"
        assert item["sha256"] == downloader.file_hash(tmp_path / item["file"])
    # A crash before the manifest commit can safely reconstruct identical CSVs.
    again = downloader.extract_document(_archive(), output_dir=tmp_path,
        document_url=api.API_BASE + "/archive/document/1", posting_date="2026-02-03T12:00:00",
        start=date(2025, 12, 5), end=date(2025, 12, 5))
    assert again == result


def test_generation_is_opt_in_and_required_fields_fail_closed(tmp_path: Path) -> None:
    result = downloader.extract_document(_archive(), output_dir=tmp_path,
        document_url=api.API_BASE + "/archive/document/2", posting_date="2026-02-03",
        start=date(2025, 12, 5), end=date(2025, 12, 6), include_generation=True)
    assert {f["kind"] for f in result["files"]} == {"CLR", "ESR", "GEN"}
    with pytest.raises(ValueError, match="missing columns"):
        downloader.extract_document(_archive(missing_base_point=True), output_dir=tmp_path,
            document_url=api.API_BASE + "/archive/document/3", posting_date="2026-02-03",
            start=date(2025, 12, 5), end=date(2025, 12, 6))


def test_dry_run_does_not_authenticate_or_touch_output(tmp_path, monkeypatch) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("No network/credentials allowed")
    monkeypatch.setattr(downloader, "ErcotClient", forbidden)
    monkeypatch.setattr(downloader.getpass, "getpass", forbidden)
    target = tmp_path / "not-created"
    assert downloader.main(["--dry-run", "--end", "2026-01-01", "--out-dir", str(target)]) == 0
    assert not target.exists()


def test_archive_pagination_uses_posting_dates_and_includes_later_corrections() -> None:
    calls = []
    class FakeClient:
        def get(self, params, **kwargs):
            calls.append((dict(params), kwargs))
            return {"_meta": {"totalPages": 2}, "archives": [{"postDatetime": f"2026-03-{params['page']:02}"}]}
    result = downloader.list_archives(FakeClient(), date(2025, 12, 5), through=date(2026, 9, 23))
    assert len(result) == 2
    assert calls[0][0]["postDatetimeFrom"] == "2026-02-03T00:00:00"
    assert calls[0][0]["postDatetimeTo"] == "2026-09-24T00:00:00"
    assert calls[1][0]["page"] == 2


def test_authenticated_archive_transport_and_untrusted_host_guard(monkeypatch) -> None:
    calls = []
    def fake_request(url, **kwargs):
        calls.append((url, kwargs))
        return b"ZIP"
    monkeypatch.setattr(api, "request_json", fake_request)
    client = api.ErcotClient("not-real", "not-real", "not-real", 0)
    client.token = "test-token"
    assert client.get({}, endpoint=api.API_BASE + "/archive/document/1", binary=True) == b"ZIP"
    assert calls[0][1]["binary"]
    assert calls[0][1]["headers"]["Accept"] == "application/json"
    with pytest.raises(ValueError, match="within ERCOT"):
        client.get({}, endpoint="https://untrusted.example/archive/1", binary=True)
    assert len(calls) == 1


def test_complete_download_and_resume_with_fake_transport(tmp_path: Path, monkeypatch) -> None:
    calls = []
    url = api.API_BASE + "/archive/document/test"
    class FakeClient:
        def __init__(self, *args):
            pass
        def get(self, params, **kwargs):
            calls.append(kwargs)
            if kwargs.get("binary"):
                return _archive()
            return {"_meta": {"totalPages": 1}, "archives": [{
                "postDatetime": "2026-02-03T12:00:00",
                "_links": {"endpoint": {"href": url}},
            }]}
    monkeypatch.setattr(downloader, "ErcotClient", FakeClient)
    for key in ("ERCOT_USERNAME", "ERCOT_PASSWORD", "ERCOT_SUBSCRIPTION_KEY"):
        monkeypatch.setenv(key, "fake-credential-not-to-be-saved")
    argv = ["--end", "2025-12-05", "--out-dir", str(tmp_path)]
    assert downloader.main(argv) == 0
    manifest_text = (tmp_path / "manifest.json").read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    assert manifest["download_complete"]
    assert "fake-credential" not in manifest_text
    assert len(manifest["documents"][url]["files"]) == 2
    assert downloader.main(argv) == 0
    assert sum(bool(c.get("binary")) for c in calls) == 1
