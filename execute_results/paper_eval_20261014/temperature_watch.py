"""CPU Packageを30秒ごとに記録し、温度条件をJSONに保存する。"""
from __future__ import annotations

from collections import deque
import csv
from datetime import datetime
import json
from pathlib import Path
import re
import subprocess
import time
import urllib.request

HERE = Path(__file__).resolve().parent


def read_sensors():
    with urllib.request.urlopen('http://127.0.0.1:8085/data.json',timeout=5) as response:
        root=json.load(response)
    stack=[root]; cpu=None; pump=None
    while stack:
        node=stack.pop(); stack.extend(node.get('Children',[]))
        sid=node.get('SensorId',''); label=node.get('Text','')
        m=re.search(r'-?\d+(?:\.\d+)?',str(node.get('Value','')))
        if not m: continue
        val=float(m.group())
        if sid=='/intelcpu/0/temperature/26' and label=='CPU Package': cpu=val
        if sid=='/lpc/nct6798d/0/fan/5': pump=val
    if cpu is None: raise RuntimeError('CPU Package sensor is missing')
    return cpu,pump


def main():
    hist=deque(maxlen=40); failures=0; previous=None
    csv_path=HERE/'temperature.csv'
    if not csv_path.exists():
        csv_path.write_text('time,cpu_package_c,gpu_c,fan6_rpm\n',encoding='utf-8')
    while True:
        now=datetime.now().astimezone().isoformat(timespec='seconds')
        try:
            cpu,pump=read_sensors(); failures=0; hist.append(cpu)
            vals=sorted(list(hist)[-4:]); median=(vals[1]+vals[2])/2 if len(vals)==4 else None
            reason=None
            if cpu>=90: reason='CPU Package >=90 C'
            gpu=float(subprocess.check_output(['nvidia-smi','--query-gpu=temperature.gpu','--format=csv,noheader,nounits'],text=True,creationflags=subprocess.CREATE_NO_WINDOW).strip())
            if gpu>=70: reason='GPU >=70 C'
            state={'time':now,'cpu_package_c':cpu,'gpu_c':gpu,'fan6_rpm':pump,
                   'median_2min_c':median,'max_20min_c':max(hist),'hot_reason':reason,
                   'valid':True,'cool_for_restart':len(vals)==4 and max(vals)<80 and gpu<60,
                   'cpu_stop_c':90,'gpu_stop_c':70}
            with csv_path.open('a',encoding='utf-8',newline='') as f: csv.writer(f).writerow([now,cpu,gpu,pump])
        except Exception as exc:
            failures+=1;state={'time':now,'valid':False,'sensor_failures':failures,'error':str(exc),'cool_for_restart':False}
        path=HERE/'temperature_status.json';tmp=path.with_suffix('.tmp')
        tmp.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(path)
        event=state.get('hot_reason') or ('sensor_missing' if not state['valid'] and failures>=3 else None)
        if event!=previous:
            with (HERE/'events.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps({'time':now,'event':'temperature_condition','reason':event,'cpu_package_c':state.get('cpu_package_c')},ensure_ascii=False)+'\n')
            previous=event
        time.sleep(30)


if __name__=='__main__': main()
