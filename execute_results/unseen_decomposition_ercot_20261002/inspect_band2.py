import numpy as np
import common
common.configure()
cases = {c[0]: c for c in common.day_cases()}
for day_index, s, ts in ((0, 10, (123, 124, 126)), (1, 4, (108, 109))):
    _, entry, fb = cases[day_index]
    cmd = fb['activation_scenario_payload'][s]
    up = np.asarray(cmd['up_proxy'], float); dn = np.asarray(cmd['down_proxy'], float)
    for t in ts:
        b = t // 6
        reg = fb['down_plan'][b] * dn[t] - fb['up_plan'][b] * up[t]
        print(entry['service_date'], s, t, f"up_proxy={up[t]:.3e} down_proxy={dn[t]:.3e} regulation_kw={reg:.3e}")
