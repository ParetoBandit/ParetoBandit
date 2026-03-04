import json
from scipy.integrate import trapezoid
with open('experiments/03_figure/results/prequential_results.json') as f:
    d = json.load(f)
k2 = d['K2']

def pareto_hull(c, r):
    pts = sorted(zip(c, r))
    hc, hr = [], []
    mr = -float('inf')
    for c_i, r_i in pts:
        if r_i > mr:
            hc.append(c_i)
            hr.append(r_i)
            mr = r_i
    return hc, hr

def interp(hc, hr, qc):
    res = []
    for q in qc:
        if q <= hc[0]: res.append(hr[0])
        elif q >= hc[-1]: res.append(hr[-1])
        else:
            for i in range(len(hc)-1):
                if hc[i] <= q <= hc[i+1]:
                    f = (q - hc[i]) / (hc[i+1]-hc[i])
                    res.append(hr[i] + f*(hr[i+1]-hr[i]))
                    break
    return res

import numpy as np

cost_range = k2.get('pareto_auc_dev_selected', k2.get('pareto_auc', {})).get('cost_range')
if cost_range:
    cost_min, cost_max = cost_range
else:
    cost_min, cost_max = 0.0, 0.02
grid = np.linspace(cost_min, cost_max, 500)

for name in ['banditgpt_pareto', 'coldstart_pareto', 'tabula_rasa_pareto']:
    data = k2[name]
    hc, hr = pareto_hull([p['mean_cost'] for p in data], [p['mean_reward'] for p in data])
    y = interp(hc, hr, grid)
    print(f"{name} AUC:", trapezoid(y, grid))

supervised = k2.get("supervised", {})
for kind, sv in supervised.items():
    print(f"Supervised {kind}: R={sv['reward']:.4f} C=${sv['cost']:.6f}")
