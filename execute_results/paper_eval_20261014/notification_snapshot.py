"""未通知の完了・異常だけを短いJSONで返す。通常進行は出力しない。"""
import json
from pathlib import Path
from datetime import datetime

HERE=Path(__file__).resolve().parent
cursor=HERE/'notification_cursor.json'
saved=json.loads(cursor.read_text(encoding='utf-8')) if cursor.exists() else {'line':0}
lines=(HERE/'events.jsonl').read_text(encoding='utf-8').splitlines()
events=[json.loads(x) for x in lines[saved['line']:]]
meaningful=[]
last_report=saved.get('last_report',{})
for e in events:
    if (e['event']=='temperature_pause' and e.get('experiment')=='afm_port_aemo6'
            and (HERE.parent/'prior_marl_20261014'/'EXCLUDED_FROM_PAPER.json').exists()
            and e.get('cpu_c',100) is not None and e.get('cpu_c',100)<90):
        # 原著の全処理を含まない試作の明示的な停止。高温停止の通知には含めない。
        continue
    if e['event'] in ('coordinator_error','temperature_pause','temperature_monitor_unavailable','experiment_complete','experiment_failed','algorithm_attention'):
        meaningful.append(e)
    elif e['event']=='reports_updated':
        counts={k:e.get(k,0) for k in ('complete_evaluations','loss_complete','complete_bess_sweeps')}
        if counts!=last_report:
            meaningful.append({k:e.get(k,0) for k in ('time','event','complete_evaluations','loss_complete','complete_bess_sweeps')})
            last_report=counts
if meaningful:
    print(json.dumps(meaningful,ensure_ascii=False))
cursor.write_text(json.dumps({'line':len(lines),'time':datetime.now().astimezone().isoformat(),'last_report':last_report}),encoding='utf-8')
