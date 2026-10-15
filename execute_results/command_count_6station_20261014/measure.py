"""本線と同じEV候補・未見40組で、設計指令64本の入札を測る。"""
from pathlib import Path
import importlib.util
import json
import os
import sys
import types

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SW=ROOT/'execute_results/resbid_6station_20261012'
BASE='wr_cap77wd_min70_margin7_validation_5_6station_128cmd_3ev_aemo_plan_deviation'
VARIANT=BASE.replace('_128cmd_3ev_', '_64cmd_low2max_')
os.environ['CUDA_VISIBLE_DEVICES']='-1'
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(SW))


def build():
    import stage_worker
    sys.modules['Config']=types.ModuleType('Config')
    stage_worker.configure('aemo')
    del sys.modules['Config']
    os.environ['EVMA_LOWER_TRAIN_UPPER_BID_ACTIVATION_SCENARIOS']='64'
    os.environ['EVMA_LOWER_TRAIN_TEST_BID_BANK_DIR']=str(ROOT/'execute_results/bid_banks'/VARIANT)
    from training.bid_bank import BidBank
    from training import lower_bid_training as lbt
    bank=BidBank(ROOT/'execute_results/bid_banks'/BASE)
    original={str(x['service_date']):x for x in (bank.load_entry(e) for e in bank.entries)}
    def fixed_evs(*,count,seed,arrival_probs,label,service_date,workers):
        base=original[str(service_date)]
        if int(base['forecast_seed'])!=int(seed):
            raise ValueError('EV forecast seed differs from baseline')
        return base['bid_ev_scenario_bank'],base['ev_scenario_selection'],None
    lbt._selected_ev_scenario_bank=fixed_evs
    activation=lbt._activation_scenarios_for_day
    def subset(*args,**kwargs):
        if kwargs.get('n_scenarios')==64:
            full=dict(kwargs,n_scenarios=128)
            payloads,mode=activation(*args,**full)
            return payloads[:64],mode
        return activation(*args,**kwargs)
    lbt._activation_scenarios_for_day=subset
    import tools.build_training_bid_bank as builder
    sys.argv=['build_training_bid_bank.py','--split','test','--days','5','--train-split-count','25',
              '--paired-train-days','25','--paired-test-days','5','--workers','1','--scenario-workers','1',
              '--output-dir',str(ROOT/'execute_results/bid_banks'/VARIANT)]
    builder.main()


def evaluate():
    p=ROOT/'execute_results/paper_eval_20261014/eval_episode.py'
    spec=importlib.util.spec_from_file_location('comparison_eval',p)
    evaluator=importlib.util.module_from_spec(spec);spec.loader.exec_module(evaluator)
    evaluator.SW_PATTERN=str(SW)
    evaluator._setup('aemo')
    for label,name in [('128',BASE),('64',VARIANT)]:
        evaluator._CACHE['test_bank']=ROOT/'execute_results/bid_banks'/name
        out=HERE/f'result_{label}.jsonl'
        rows=[json.loads(x) for x in out.read_text(encoding='utf-8').splitlines()] if out.exists() else []
        done={(r['day'],r['command'],r['ev_seed']) for r in rows}
        for d in range(5):
            for c in range(4):
                for e in range(2):
                    if (d,c,e) in done:continue
                    result=evaluator.run_episode(('aemo','lp',d,c,e,4))
                    with out.open('a',encoding='utf-8') as f:f.write(json.dumps(result,ensure_ascii=False)+'\n')
                    print(label,d,c,e,'feasible',result['feasible'],flush=True)
    report={}
    for label in ('64','128'):
        rows=[json.loads(x) for x in (HERE/f'result_{label}.jsonl').read_text(encoding='utf-8').splitlines()]
        if len(rows)!=40 or len({(r['day'],r['command'],r['ev_seed']) for r in rows})!=40:
            raise RuntimeError('expected 40 distinct pairs')
        report[label]={'sets':40,'capacity_kw_block_day':sum(r['offered_capacity_kw_block'] for r in rows)/40,
                       'feasible':sum(r['feasible'] is True for r in rows),
                       'infeasible':sum(r['feasible'] is False for r in rows),
                       'unknown':sum(r['feasible'] is None for r in rows)}
    (HERE/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':
    if sys.argv[1]=='build':build()
    elif sys.argv[1]=='evaluate':evaluate()
    else:raise ValueError(sys.argv[1])
