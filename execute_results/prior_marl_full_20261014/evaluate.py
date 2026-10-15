"""Final fixed 40 pairs: own bidding and the common-bid control ablation."""
import argparse
import json
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from settings import configure, HERE, ROOT


def final_row(row, day, command, ev, model, episode):
    import numpy as np
    trace = row['traces']
    err = np.abs(trace['residual_kw'])
    tol = np.asarray(trace['tol_kw'])
    tracking = np.asarray(trace['tracking'], bool)
    up, down = np.asarray(trace['up_kw']), np.asarray(trace['down_kw'])
    baseline = np.asarray(trace['baseline_kw'])
    live = (up > 0) | (down > 0)
    passed = err <= tol + 1e-6
    blocks = [dict(k=k, up=float(up[k]), down=float(down[k]),
                   **{'pass': bool(np.mean(passed[k * 6:(k + 1) * 6]) >= .9)},
                   max_err_kw=float(err[k * 6:(k + 1) * 6].max()),
                   band_kw=float(np.min(tol[k * 6:(k + 1) * 6])), baseline_kw=float(baseline[k]),
                   baseline_step_kw=float(abs(baseline[k] - baseline[k - 1])) if k and live[k - 1] else None,
                   first_step_error_kw=float(err[k * 6]),
                   mae_kw=float(np.mean(err[k * 6:(k + 1) * 6]))) for k in range(48) if live[k]]
    target = np.asarray(trace['target_kw'])
    regulation = target - np.repeat(baseline, 6)
    award = np.where(regulation < -1e-6, np.repeat(up, 6), np.repeat(down, 6))
    mask = np.repeat(live, 6)
    return dict(row, task='aemo', mode='marl', day=day, command=command, ev_seed=ev,
                ev_environment_seed=4000000 + 1000 * day + 10 * command + ev,
                model=model, episode=episode, soc_hit_pct=row['local_rate'],
                step_pass_pct=float(100 * np.mean(passed[tracking])) if tracking.any() else None,
                block_pass_pct=100 * np.mean([b['pass'] for b in blocks]) if blocks else None,
                blocks_assessed=len(blocks), blocks_failed=sum(not b['pass'] for b in blocks),
                mean_error_to_award_pct=float(100 * np.sum(err[mask]) / max(np.sum(award[mask]), 1e-9)),
                offered_capacity_kw_block=row['capacity_kw_block'], blocks=blocks)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    args = p.parse_args()
    configure('aemo')
    import torch
    from networks import AFMLearner, Specification
    from driver import inputs, episode, aggregate
    from environment.normalize import configure_observation_normalization
    from training.lower_bid_training import _activation_scenarios_for_day
    run = Path(args.run)
    saved = torch.load(run / 'policy_best.pth', map_location='cpu', weights_only=False)
    if saved['learned_episodes'] < 1000:
        raise ValueError('publication checkpoint requires >=1000 learned episodes')
    learner = AFMLearner(Specification(**saved['specification']))
    for name, value in saved['networks'].items():
        getattr(learner, name).load_state_dict(value)
    configure_observation_normalization(json.loads((run / 'observation_normalization.json').read_text()))
    _, payloads, _, _ = inputs('aemo')
    common_out = ROOT / 'execute_results/paper_eval_20261014/prior6_aemo_v2'
    own_out = run / 'final_learned_bid'
    for common, output in [(False, own_out), (True, common_out)]:
        output.mkdir(parents=True, exist_ok=True)
        file = output / 'result_aemo_marl_4cmd_2ev.jsonl'
        rows = [json.loads(line) for line in file.read_text().splitlines()] if file.exists() else []
        done = {(row['day'], row['command'], row['ev_seed']) for row in rows}
        for day, payload in enumerate(payloads):
            commands, _ = _activation_scenarios_for_day(payload['service_date'], payload['forecast_seed'] + 512209,
                                                       n_scenarios=4, scenario_partition='holdout')
            for command, activation in enumerate(commands):
                for ev in range(2):
                    if (day, command, ev) in done:
                        continue
                    row = episode(learner, payload, activation, seed=4000000 + 1000 * day + 10 * command + ev,
                                  train=False, common_bid=common)
                    row = final_row(row, day, command, ev, run.name, saved['episode'])
                    with file.open('a', encoding='utf-8') as f:
                        f.write(json.dumps(row) + '\n')
                    rows.append(row)
                    print('final', common, day, command, ev, row['global_rate'], row['local_rate'], flush=True)
        if len(rows) != 40 or len({(r['day'], r['command'], r['ev_seed']) for r in rows}) != 40:
            raise ValueError('final evaluation requires 40 unique pairs')
        (output / 'summary.json').write_text(json.dumps(aggregate(rows), indent=2))
        if common:
            verification = json.loads((HERE / 'implementation_status.json').read_text())
            verification.update(model=run.name, episode=saved['episode'], learned_episodes=saved['learned_episodes'])
            if not verification.get('verified'):
                raise ValueError('method tests not verified')
            (output / 'method_verification.json').write_text(json.dumps(verification, indent=2))


if __name__ == '__main__':
    main()
