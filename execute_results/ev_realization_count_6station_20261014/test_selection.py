"""Check the paired forecast candidates without performing a bid solve."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import measure
measure.configure('aemo')


def test_four_realizations_extend_the_main_three_for_all_dates_and_markets():
    for task in ('aemo', 'ercot'):
        records, proof = measure.prepare_forecasts(task)
        assert len(records) == len(proof) == 5
        for row in proof:
            assert row['candidate_count'] == 128
            assert len(row['three_indices']) == 3 and len(row['four_indices']) == 4
            assert set(row['three_indices']).issubset(row['four_indices'])
            assert row['baseline_physical_forecasts_verified']
