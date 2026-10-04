#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/../.."
L=execute_results/ft_sweep/bid_scale_sweep.log
while ! grep -q "=== スイープ完了" "$L" 2>/dev/null; do
  AGE=$(( $(date +%s) - $(stat -c %Y "$L") ))
  if [ "$AGE" -gt 1200 ]; then
    echo "[$(date '+%F %T')] スイープ無更新 ${AGE}s"; tail -3 "$L"; exit 1
  fi
  sleep 60
done
echo "[$(date '+%F %T')] ===== 入札倍率スイープ 完了 ====="
python - <<'PY'
import csv, os
print(f"{'倍率':>6s} {'ロールアウト':>10s} {'追従':>8s} {'上げ合格':>9s} {'下げ合格':>9s} {'MAE kW':>9s} {'SoC':>7s}")
for tag, s in (("bs1.00","1.00"),("bs0.80","0.80"),("bs0.65","0.65"),("bs0.50","0.50")):
    p = f"execute_results/ft_sweep/{tag}/summary_overall.csv"
    if not os.path.exists(p):
        print(f"{s:>6s} {'(なし)':>10s}"); continue
    r = list(csv.DictReader(open(p, encoding="utf-8")))[-1]
    print(f"{s:>6s} {r['rollouts']:>10s} "
          f"{float(r['global_tracking_rate'])*100:7.2f}% "
          f"{float(r['up_pass_rate'])*100:8.2f}% "
          f"{float(r['down_pass_rate'])*100:8.2f}% "
          f"{float(r['post_bess_mae_kw']):9.2f} "
          f"{float(r['soc_hit_rate'])*100:6.1f}%")
PY
