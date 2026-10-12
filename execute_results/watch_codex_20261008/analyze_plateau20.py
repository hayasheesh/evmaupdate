"""保存済みCSVとTensorBoardから20局の学習停滞を集計する。"""
import csv
import json
from pathlib import Path
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
NAMES = {'aemo': 'allin_aemoplan_AB_20station_20261007_042318',
         'ercot': 'allin_ercotplan_AB_20station_20261007_042436'}
TAGS = ['Gradient/actor_source_local_raw_mean','Gradient/actor_source_global_raw_mean',
        'Gradient/actor_source_cos_mean','Gradient/actor_source_global_ratio_mean',
        'Gradient/global_critic_raw','Clipping/global_critic','Loss/global_critic',
        'Loss/local_critic_mean','GlobalCritic/td_error_abs_mean','Q/global','Q/local_mean',
        'Training/buffer_fill_rate','Training/epsilon','Training/ou_noise_scale',
        'Reward/local','Reward/global','Reward/local_departure','Reward/local_shaping']

def main():
    result = {}
    for market, name in NAMES.items():
        run = ROOT/'archive'/name
        scalars = {}
        for f in (run/'performance').glob('events.out.tfevents.*'):
            acc = EventAccumulator(str(f),size_guidance={'scalars':0})
            acc.Reload()
            for tag in acc.Tags()['scalars']:
                scalars.setdefault(tag,{}).update({x.step:x.value for x in acc.Scalars(tag)})
        hist = json.loads((run/'results/test_history.json').read_text())
        with (run/'results/test_performance_metrics.csv').open(encoding='utf-8-sig',newline='') as f:
            rows = list(csv.DictReader(f))
        tests = [{'episode':ep,'tracking':float(r['Dispatch_Tracking_Rate_%']),
                  'soc':float(r['SoC_Hit_Rate_%']),'mae':float(r['Raw_Actor_MAE_kW'])}
                 for ep,r in zip(hist['episodes'],rows)]
        for t in tests:t['sum']=t['tracking']+t['soc']
        best = max((t for t in tests if t['episode']>=1000),key=lambda t:t['sum'])
        item = {'best_after1000':best,'latest_test':tests[-1],'windows':{},'stations':{}}
        for lo,hi in [(900,1099),(1100,1299),(1300,1499),(1500,1699)]:
            def mean(tag):
                v=[v for ep,v in scalars.get(tag,{}).items() if lo<=ep<=hi]
                return float(np.mean(v)) if v else None
            block={tag:mean(tag) for tag in TAGS}
            block['actor_raw_mean']=float(np.mean([mean(f'Gradient/actor_raw_agent{i}') for i in range(1,21)]))
            block['actor_clip_fraction']=float(np.mean([mean(f'Clipping/actor_agent{i}') for i in range(1,21)]))
            block['local_critic_clip_fraction']=float(np.mean([mean(f'Clipping/local_critic_agent{i}') for i in range(1,21)]))
            tt=[t for t in tests if lo<=t['episode']<=hi]
            block['test_mean']={k:float(np.mean([t[k] for t in tt])) for k in ['tracking','soc','sum','mae']}
            block['test_count']=len(tt)
            item['windows'][f'{lo}-{hi}']=block
        for i in range(1,21):
            def tail(tag):
                vals=[v for ep,v in scalars.get(tag,{}).items() if ep>=1500]
                return float(np.mean(vals)) if vals else None
            item['stations'][str(i)]={'local_raw':tail(f'Gradient/actor_source_local_raw_agent{i}'),
                                      'global_raw':tail(f'Gradient/actor_source_global_raw_agent{i}'),
                                      'cos':tail(f'Gradient/actor_source_cos_agent{i}'),
                                      'actor_raw':tail(f'Gradient/actor_raw_agent{i}'),
                                      'clip_fraction':tail(f'Clipping/actor_agent{i}')}
        item['nonfinite_recorded_scalars']=sum(not np.isfinite(v) for tag in scalars.values() for v in tag.values())
        item['last_scalar_episode']=max(ep for tag in scalars.values() for ep in tag)
        result[market]=item
    (OUT/'plateau20_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    for m,r in result.items():
        print(m,json.dumps({k:v for k,v in r.items() if k!='stations'},ensure_ascii=False))
if __name__=='__main__':main()
