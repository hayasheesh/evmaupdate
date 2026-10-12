"""評価に使った holdout の指令（日×指令）ごとに、指令の偏りを数え、Liu/Lyu の結果と並べる。

regulation = target - baseline（充電を正）。偏りの電力量 = Σ regulation × 5分（kWh、正は基準より多く充電させる向き）。
同じ向きが続いた最長の時間と、計画が使った指令の分布（学習用の指令）の平均との差も出す。
"""
import json
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
import run_port


def main(task):
    sw = run_port._setup(task)
    import liu_lyu_port as L
    from training.bid_bank import BidBank
    from training.lower_bid_training import _activation_scenarios_for_day, build_fixed_upper_bid_training_episode
    _t, test_bank = sw.banks(task)
    bank = BidBank(test_bank)
    res = {}
    for name in ('fixed',):
        p = HERE / f'result_{task}_{name}_force0_4cmd_2ev.jsonl'
        for l in p.read_text(encoding='utf-8').splitlines():
            r = json.loads(l)
            res.setdefault((r['day'], r['command']), []).append(r)
    for d in range(5):
        base_bid = dict(bank.load_entry(bank.entries[d]))
        payloads, mode = _activation_scenarios_for_day(base_bid.get('service_date'), int(base_bid.get('forecast_seed', 0)) + 512_209,
                                                       n_scenarios=4, scenario_partition='holdout')
        fb = dict(base_bid, activation_scenario_payload=list(payloads), activation_scenarios=len(payloads), activation_mode=mode)
        probs = L.scenario_probabilities(base_bid['design_activation_scenario_payload'])
        exp_y = probs @ L.SCEN
        base = np.repeat(np.asarray(fb['baseline_plan'], float), 6)
        for c in range(4):
            target, tol, arr, _ = build_fixed_upper_bid_training_episode(fb, c)
            reg = np.asarray(target, float) - base
            y = np.asarray(payloads[c]['down_proxy'], float)[:288] - np.asarray(payloads[c]['up_proxy'], float)[:288]
            sign = np.sign(np.round(y, 6))
            longest, cur, prev = 0, 0, 0
            for s in sign:
                cur = cur + 1 if (s != 0 and s == prev) else (1 if s != 0 else 0)
                prev = s
                longest = max(longest, cur)
            blk_mean = y.reshape(48, 6).mean(axis=1)
            rr = res.get((d, c), [])
            print(f"{task} day{d} {base_bid['service_date']} cmd{c} {payloads[c]['source_bmu']} {payloads[c]['source_date']}: "
                  f"energy_bias={reg.sum() / 12:+.0f}kWh  mean_y={y.mean():+.3f} (planned {exp_y.mean():+.3f})  "
                  f"longest_same_sign={longest * 5}min  |y|>0.9 steps={int((np.abs(y) > 0.9).sum())}  "
                  f"failed_blocks={[r['blocks_failed'] for r in rr]} soc={[round(r['soc_hit_pct']) for r in rr]}")


main(sys.argv[1] if len(sys.argv) > 1 else 'aemo')
