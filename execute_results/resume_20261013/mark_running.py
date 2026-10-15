"""再開した学習の <task>_run_status.json を「実行中」に戻す（見張りが前の complete を拾い直さないように）。

    python mark_running.py <run_status.json> <pid> <run_dir>
"""
import json
import sys
from datetime import datetime
from pathlib import Path

path, pid, run_dir = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
state = json.loads(path.read_text(encoding='utf-8'))
previous = state.pop('result', None)
state['stages']['pretrain'] = {'state': 'running', 'pid': pid, 'started_at': datetime.now().isoformat(timespec='seconds'),
                               'resumed_from': run_dir, 'previous_result': previous}
path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print('marked running', path.name)
