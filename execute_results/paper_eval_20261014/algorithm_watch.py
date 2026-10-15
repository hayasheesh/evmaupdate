"""先行方式の進行・異常を差分だけで確認する。通常進行は通知しない。"""
from pathlib import Path
from datetime import datetime
import json
import re
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def check(process_rows,event):
    state_file=HERE/'algorithm_watch_state.json'
    previous=json.loads(state_file.read_text(encoding='utf-8')) if state_file.exists() else {}
    alive={int(x['ProcessId']) for x in process_rows}
    report={};new_state={}
    for folder in sorted((ROOT/'execute_results').glob('prior_marl*')):
        status_file=folder/'run_status.json'
        if not status_file.exists():continue
        if (folder/'EXCLUDED_FROM_PAPER.json').exists():
            report[folder.name]={'status':'excluded_partial_prototype'}
            continue
        status=json.loads(status_file.read_text(encoding='utf-8'))
        stage=status.get('stage','')
        key=folder.name+':'+str(status.get('model',''))
        old=previous.get(key,{})
        item=dict(old)
        snapshot={'stage':stage,'model':status.get('model'),'pid':status.get('pid')}
        log=Path(status['log']) if status.get('log') else folder/'logs'/f"{status.get('model','')}.log"
        if stage.startswith('training'):
            dead=int(status.get('pid',-1)) not in alive
            item['dead_checks']=old.get('dead_checks',0)+1 if dead else 0
            if item['dead_checks']==2:
                event('algorithm_attention',experiment=folder.name,reason='training process missing',pid=status.get('pid'))
            if log.exists():
                size=log.stat().st_size
                start=old.get('offset',0)
                if start>size:start=0
                with log.open('rb') as f:
                    f.seek(max(start,size-65536));chunk=f.read().decode('utf-8',errors='replace')
                item['offset']=size
                anomalies=re.findall(r'(?m)^.*(?:FloatingPointError|Traceback \(most recent call last\)|(?:loss|gradient)[^\r\n]*\b(?:nan|inf)\b).*$',chunk)
                if anomalies:
                    event('algorithm_attention',experiment=folder.name,reason='nonfinite loss/gradient or exception',evidence=anomalies[-1][:300])
                tests=list(re.finditer(r'\[TEST(\d+) summary\].*?SoC actor/physical=([\d.]+)/[\d.]+%.*?MARL/Central=([\d.]+)/[\d.]+%',chunk))
                if tests:
                    t=tests[-1];point={'checkpoint_episode':int(t[1]),'local':float(t[2]),'global':float(t[3])}
                    point['sum']=point['local']+point['global'];item['latest_validation']=point
                snapshot['latest_validation']=item.get('latest_validation')
                stale=time.time()-log.stat().st_mtime>900 and not dead
                if stale and not old.get('stale_alert'):
                    event('algorithm_attention',experiment=folder.name,reason='training log silent for 15 minutes',silence_seconds=round(time.time()-log.stat().st_mtime))
                item['stale_alert']=stale
        report[folder.name]=snapshot;new_state[key]=item
    state_file.write_text(json.dumps(new_state,ensure_ascii=False,indent=2),encoding='utf-8')
    p=HERE/'algorithm_health.json';tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps({'time':datetime.now().astimezone().isoformat(),'methods':report},ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(p)
