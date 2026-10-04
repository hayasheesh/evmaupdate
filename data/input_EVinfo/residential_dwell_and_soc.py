"""Convert the Norway residential charging reports into a dwell/required-SoC table.

Companion to `neededsoc_and_dwell.py`, which does the same job for the ACN
workplace sessions. The output columns and the two capacity bases are identical,
so the two tables are interchangeable as an EV profile pool.

Source: Dataset1_charging_reports.csv from Zenodo 10.5281/zenodo.12730566
(CC BY 4.0). Only the two locations that supply the residential arrival curves
(OSL_S, TRO_R) are used, so the dwell distribution and the arrival curve come
from the same sites.

Dwell and requested energy are taken from the same session row, preserving the
empirical relation between stay length and energy need that EVEnv relies on
when it samples both from one profile index.
"""

import csv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
SOURCE = (
    BASE_DIR / "external_sources" / "norway_residential"
    / "Dataset1_charging_reports.csv"
)
# The two sites behind Arrival__RESIDENTIAL NORWAY _ OSL_S.csv and _ TRO_R.csv.
LOCATIONS = ("OSL_S", "TRO_R")
ROUND_MIN = 5
# Same two bases as the workplace table: the legacy 50 kWh denominator EVEnv
# still defaults to, and the corrected 100 kWh one matching EnvConfig.EV_CAPACITY.
OUTPUTS = {
    50.0: BASE_DIR / "RESIDENTIAL_NORWAY_neededsocanddwelltime.csv",
    100.0: BASE_DIR / "RESIDENTIAL_NORWAY_neededsocanddwelltime_cap100.csv",
}


def _decimal(value):
    """Parse one field written with a decimal comma, as the source file uses."""
    try:
        return float(str(value).strip().replace(",", "."))
    except (TypeError, ValueError):
        return None


def load_sessions():
    """Return (dwell_5min_units, energy_kwh) for every usable session row."""
    rows = []
    with open(SOURCE, newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle, delimiter=";"):
            if str(row.get("location", "")).strip() not in LOCATIONS:
                continue
            hours = _decimal(row.get("connection_time"))
            energy = _decimal(row.get("energy_session"))
            if hours is None or energy is None or hours <= 0.0 or energy < 0.0:
                continue
            units = int(round(hours * 60.0 / ROUND_MIN))
            if units <= 0:
                continue
            rows.append((units, energy))
    if not rows:
        raise ValueError(f"No usable residential sessions in {SOURCE}")
    return rows


def write_table(rows, capacity_kwh, path):
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ev_id", "connection_minutes_5min", "required_soc_percent"])
        for ev_id, (units, energy) in enumerate(rows, start=1):
            percent = min(max(energy / capacity_kwh * 100.0, 0.0), 100.0)
            writer.writerow([ev_id, units, round(percent, 4)])
    return path


def main():
    rows = load_sessions()
    dwell_hours = [u * ROUND_MIN / 60.0 for u, _ in rows]
    energies = [e for _, e in rows]
    print(f"Sessions from {'/'.join(LOCATIONS)}: {len(rows)}")
    print(f"  dwell   mean {sum(dwell_hours)/len(rows):.2f} h, "
          f"max {max(dwell_hours):.2f} h")
    print(f"  energy  mean {sum(energies)/len(energies):.2f} kWh, "
          f"max {max(energies):.2f} kWh")
    for capacity_kwh, path in OUTPUTS.items():
        write_table(rows, capacity_kwh, path)
        clipped = sum(1 for e in energies if e / capacity_kwh * 100.0 > 100.0)
        print(f"Wrote: {path.name}  basis {capacity_kwh:g} kWh  "
              f"clipped at 100% {clipped} rows ({100.0*clipped/len(rows):.2f}%)")


if __name__ == "__main__":
    main()
