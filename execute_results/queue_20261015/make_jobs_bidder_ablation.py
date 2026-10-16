"""C4（最低入札量と余裕を振る）と、手当てを1つずつ外す比較の順番待ち（Windows、6 station、2026-10-16）。

    python make_jobs_bidder_ablation.py      jobs_winabl1〜4.json を書く

各変種で、検証用の入札（5日）を作り（table.py build）、未見通過率を測り（table.py unseen）、本線の行動器で40組を測る
（run_eval.py、全情報のLP・ルールベース・行動器・行動器＋強制充電）。EVを無作為に選ぶ入札（sel_random0〜2）は既にあるので、40組だけ測る。
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = {'aemo': 'wr_cap77wd_aemoplan_6station_resbid_tanh_ls717_20261012_102306',
        'ercot': 'wr_cap77wd_ercotplan_6station_resbid_tanh_ls717_20261013_023145'}
SEL = 'execute_results/selection_table_6station_20261015'


def build(task, v):
    return [{'id': f'abl_{task}_build_{v}', 'cwd': SEL, 'env': {'CUDA_VISIBLE_DEVICES': '-1'}, 'priority': 'below_normal',
             'args': ['table.py', 'build', task, v]},
            {'id': f'abl_{task}_unseen_{v}', 'cwd': SEL, 'env': {'CUDA_VISIBLE_DEVICES': '-1'}, 'priority': 'below_normal',
             'args': ['table.py', 'unseen', task, v, '6']}]


def evaluate(task, v):
    return [{'id': f'abl_{task}_eval_{v}', 'cwd': '.', 'env': {'CUDA_VISIBLE_DEVICES': '-1', 'BIDDER_VARIANT': v},
             'priority': 'below_normal',
             'requires': [f'archive/{RUNS[task]}/results/TEST2000/actor_5_ep2000.pth'],
             'args': ['execute_results/paper_eval_20261014/run_eval.py', '--task', task,
                      '--stage-dir', '{ROOT}/execute_results/bidder_ablation_6station_20261016',
                      '--run-dir', '{ROOT}/archive/' + RUNS[task], '--episode', '2000',
                      '--output', '{ROOT}/execute_results/paper_eval_20261014/' + f'abl6_{task}_{v}',
                      '--modes', 'lp', 'rule', 'marl', 'marl_force', '--workers', '4']}]


def chain(task, v):
    return build(task, v) + evaluate(task, v)


QUEUES = {
    'winabl1': [x for v in ('q1', 'w0', 'q35', 'q180') for x in chain('aemo', v)],
    'winabl2': [x for v in ('m0',) for x in chain('aemo', v)]
               + [x for r in range(3) for x in evaluate('aemo', f'sel_random{r}')]
               + [x for v in ('q110', 'm14') for x in chain('aemo', v)],
    'winabl3': [x for v in ('q1', 'w0', 'q35', 'q180') for x in chain('ercot', v)],
    'winabl4': [x for v in ('m0',) for x in chain('ercot', v)]
               + [x for r in range(3) for x in evaluate('ercot', f'sel_random{r}')]
               + [x for v in ('q110', 'm14') for x in chain('ercot', v)],
}
for name, jobs in QUEUES.items():
    (HERE / f'jobs_{name}.json').write_text(json.dumps(jobs, ensure_ascii=False, indent=1), encoding='utf-8')
    print(name, len(jobs), [j['id'] for j in jobs][:4])
