"""Compare 3/4 selected EV realizations from the same 128 forecast candidates."""
from pathlib import Path
import importlib.util
import json
import os
import pickle
import sys
import types

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SW = ROOT / 'execute_results/resbid_6station_20261012'
POOL = ROOT / 'execute_results/bid_loss_6station_20261013/data'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SW))


def paths(task):
    import stage_worker
    baseline = stage_worker.banks(task)[1]
    variant = baseline.with_name(baseline.name.replace('_3ev_', '_4ev_low3max_'))
    return baseline, variant


def configure(task, four=False):
    import stage_worker
    # Apply the experiment's environment before importing the real Config.
    if 'Config' in sys.modules:
        raise RuntimeError('configure must run once in a fresh worker')
    sys.modules['Config'] = types.ModuleType('Config')
    stage_worker.configure(task)
    del sys.modules['Config']
    if four:
        os.environ['EVMA_LOWER_TRAIN_UPPER_BID_EV_SCENARIO_SELECTION'] = 'low_connection_3_max_count'
        os.environ['EVMA_LOWER_TRAIN_TEST_BID_BANK_DIR'] = str(paths(task)[1])
    import Config
    import torch
    torch.set_num_threads(1)
    assert not torch.cuda.is_available()
    assert Config.Q_MIX_GLOBAL_WEIGHT == .5 and Config.GRAD_CLIP_MAX == 5


def ev_signature(ev):
    # EV labels are solver identifiers; all physical fields are compared.
    return tuple((key, value) for key, value in sorted(vars(ev).items()) if key != 'ev_id')


def prepare_forecasts(task):
    from training.bid_bank import BidBank
    from training import lower_bid_training as lbt
    bank = BidBank(paths(task)[0])
    records, proof = {}, []
    for day, entry in enumerate(bank.entries):
        base = bank.load_entry(entry)
        cands = pickle.loads((POOL / f'candidates_day{day}.pkl').read_bytes())
        assert len(cands) == 128
        three, meta3 = lbt._select_ev_scenarios_low_connection(cands, low_picks=2, count_picks=('maximum',))
        four, meta4 = lbt._select_ev_scenarios_low_connection(cands, low_picks=3, count_picks=('maximum',))
        idx3 = [int(x['candidate_index']) for x in meta3]
        idx4 = [int(x['candidate_index']) for x in meta4]
        stored = [int(x['candidate_index']) for x in base['ev_scenario_selection']]
        if idx3 != stored:
            raise ValueError(f'{task} day{day}: baseline forecast candidate indices differ')
        if any(sorted(map(ev_signature, x)) != sorted(map(ev_signature, y))
               for x, y in zip(three, base['bid_ev_scenario_bank'])):
            raise ValueError(f'{task} day{day}: baseline physical EV forecasts differ')
        if not set(idx3).issubset(idx4) or len(set(idx4)) != 4:
            raise ValueError('four-scenario selection must contain the original three')
        records[str(base['service_date'])] = dict(base=base, four=four, meta=meta4)
        proof.append(dict(day=day, service_date=base['service_date'], forecast_seed=int(base['forecast_seed']),
                          candidate_count=128, three_indices=idx3, four_indices=idx4,
                          nested=True, baseline_physical_forecasts_verified=True))
    if len(proof) != 5:
        raise ValueError('expected five paired validation dates')
    return records, proof


def verify_commands(base, variant):
    import numpy as np
    original = base['design_activation_scenario_payload']
    new = variant['design_activation_scenario_payload']
    if len(original) != 128 or len(new) != 128:
        raise ValueError('expected 128 bid-design commands')
    for a, b in zip(original, new):
        if a.get('source') != b.get('source'):
            raise ValueError('bid-design command source differs')
        for key in ('up_proxy', 'down_proxy'):
            np.testing.assert_array_equal(a[key], b[key])
    for key in ('forecast_seed', 'service_date', 'assessment_band_fraction',
                'apply_transition_band', 'assessment_i_control_margin_kw'):
        if base[key] != variant[key]:
            raise ValueError(f'paired input differs: {key}')
    np.testing.assert_array_equal(base['arrival_probabilities_by_station'], variant['arrival_probabilities_by_station'])


def build(task):
    configure(task, four=True)
    from training import lower_bid_training as lbt
    records, proof = prepare_forecasts(task)
    (HERE / f'inputs_{task}.json').write_text(json.dumps(proof, indent=2), encoding='utf-8')

    def selected(*, count, seed, arrival_probs, label, service_date, workers):
        record = records[str(service_date)]
        if int(seed) != int(record['base']['forecast_seed']):
            raise ValueError('forecast seed changed')
        return record['four'], record['meta'], None

    lbt._selected_ev_scenario_bank = selected
    import tools.build_training_bid_bank as builder
    sys.argv = ['build_training_bid_bank.py', '--split', 'test', '--days', '5', '--train-split-count', '25',
                '--paired-train-days', '25', '--paired-test-days', '5', '--workers', '1', '--scenario-workers', '1',
                '--output-dir', str(paths(task)[1])]
    builder.main()
    from training.bid_bank import BidBank
    bank = BidBank(paths(task)[1])
    for entry in bank.entries:
        bid = bank.load_entry(entry)
        verify_commands(records[bid['service_date']]['base'], bid)
        if len(bid['bid_ev_scenario_bank']) != 4:
            raise ValueError('variant does not contain four EV realizations')


def evaluate(task):
    configure(task)
    spec = importlib.util.spec_from_file_location('count_evaluation', ROOT / 'execute_results/paper_eval_20261014/eval_episode.py')
    evaluation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluation)
    evaluation.SW_PATTERN = str(SW)
    evaluation._setup(task)
    from market.physical_lp_bidding import colgen_feasibility
    original_certify = colgen_feasibility.certify
    def certify_until_decided(*args, **kwargs):
        result = None
        for limit in (20., 120., 600.):
            result = original_certify(*args, **dict(kwargs, time_limit_s=limit))
            if result[0] is not None:
                return result
            print('LP undecided; retrying with longer time limit', limit, flush=True)
        raise RuntimeError('unseen LP case still undecided after 600s; retain results for further solving')
    colgen_feasibility.certify = certify_until_decided
    from training.bid_bank import BidBank
    base_path, variant_path = paths(task)
    base_bank, variant_bank = BidBank(base_path), BidBank(variant_path)
    for a, b in zip(base_bank.entries, variant_bank.entries):
        verify_commands(base_bank.load_entry(a), variant_bank.load_entry(b))
    reference_folder = ROOT / 'execute_results' / ('command_count_6station_20261014' if task == 'aemo' else 'command_count_ercot_6station_20261014')
    base_rows = [json.loads(x) for x in (reference_folder / 'result_128.jsonl').read_text().splitlines()]
    reference = {(r['day'], r['command'], r['ev_seed']): r for r in base_rows}
    if len(reference) != len(base_rows) or len(reference) != 40:
        raise ValueError('baseline must have 40 unique evaluation pairs')
    output = HERE / f'result_{task}_4ev.jsonl'
    rows = [json.loads(x) for x in output.read_text().splitlines()] if output.exists() else []
    done = {(r['day'], r['command'], r['ev_seed']) for r in rows}
    evaluation._CACHE['test_bank'] = variant_path
    for day in range(5):
        for command in range(4):
            for ev_seed in range(2):
                key = (day, command, ev_seed)
                if key in done:
                    continue
                result = evaluation.run_episode((task, 'lp', day, command, ev_seed, 4))
                old = reference[key]
                if result['service_date'] != old['service_date'] or result['evs'] != old['evs']:
                    raise ValueError('unseen EV/date pair changed')
                with output.open('a', encoding='utf-8') as f:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
                rows.append(result)
                print(task, day, command, ev_seed, 'feasible', result['feasible'], flush=True)
    if len(rows) != 40 or len({(r['day'], r['command'], r['ev_seed']) for r in rows}) != 40:
        raise ValueError('variant must have 40 unique evaluation pairs')
    summary = {}
    for label, cases in [('3', base_rows), ('4', rows)]:
        summary[label] = dict(sets=40, capacity_kw_block_day=sum(r['offered_capacity_kw_block'] for r in cases) / 40,
                              feasible=sum(r['feasible'] is True for r in cases),
                              infeasible=sum(r['feasible'] is False for r in cases),
                              unknown=sum(r['feasible'] is None for r in cases))
    (HERE / f'summary_{task}.json').write_text(json.dumps(summary, indent=2))


def report():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    report = {task: json.loads((HERE / f'summary_{task}.json').read_text()) for task in ('aemo', 'ercot')}
    plt.rcParams['font.family'] = ['Yu Gothic', 'Meiryo', 'sans-serif']
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for offset, label, color in [(-.18, '3', '#2a78d6'), (.18, '4', '#eb6834')]:
        x = np.arange(2) + offset
        capacity = [report[t][label]['capacity_kw_block_day'] for t in report]
        passed = [100 * report[t][label]['feasible'] / 40 for t in report]
        axes[0].bar(x, capacity, .34, label=f'EV {label}通り', color=color)
        axes[1].bar(x, passed, .34, label=f'EV {label}通り', color=color)
        for ax, values in zip(axes, (capacity, passed)):
            for xx, value in zip(x, values):
                ax.text(xx, value + (300 if ax is axes[0] else 1), f'{value:,.1f}', ha='center', fontsize=9)
    for ax in axes:
        ax.set_xticks(range(2), ['AEMO', 'ERCOT'])
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.25); ax.set_axisbelow(True)
    axes[0].set_ylabel('入札量（kW・コマ／日）')
    axes[0].set_ylim(0, max(r[k]['capacity_kw_block_day'] for r in report.values() for k in ('3', '4')) * 1.2)
    axes[1].set_ylabel('未見成立率：全情報LP（%）'); axes[1].set_ylim(0, 110)
    axes[0].legend(frameon=False)
    fig.suptitle('入札器に渡すEVの実現数：同じ予測候補128通り・設計指令128本・未見40組')
    fig.tight_layout(); fig.savefig(HERE / 'fig_ev_realization_count.png', dpi=160); plt.close(fig)
    (HERE / 'summary.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    if sys.argv[1] == 'build':
        build(sys.argv[2])
    elif sys.argv[1] == 'evaluate':
        evaluate(sys.argv[2])
    elif sys.argv[1] == 'report':
        report()
    else:
        raise ValueError(sys.argv[1])
