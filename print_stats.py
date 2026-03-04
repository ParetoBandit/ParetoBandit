import json
with open('experiments/03_figure/results/prequential_results.json') as f:
    d = json.load(f)
k2 = d['K2']
print("K2 Results:")
print("Oracle:", k2['oracle']['reward'])
supervised = k2.get('supervised', {})
for kind, sv in supervised.items():
    print(f"Supervised {kind}: R={sv['reward']:.4f} +/-{sv['std_reward']:.4f}")
print("Pareto AUC (dev-selected):", k2.get('pareto_auc_dev_selected', {}))
static = k2.get('static', {})
for m, s in static.items():
    print(f"Static {m}: R={s['reward']:.4f} C=${s['cost']:.6f}")

k10 = d.get('K10', {})
if k10:
    print("\nK10 Results:")
    print("Best static:", k10.get('best_static', {}).get('reward'))
    print("Oracle:", k10.get('oracle', {}).get('reward'))
    supervised_k10 = k10.get('supervised', {})
    for kind, sv in supervised_k10.items():
        print(f"Supervised {kind}: R={sv['reward']:.4f} +/-{sv['std_reward']:.4f}")
