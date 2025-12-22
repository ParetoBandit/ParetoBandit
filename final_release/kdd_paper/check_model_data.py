import json

with open('final_release/models.json') as f:
    data = json.load(f)['models']

for mid in ['openai/gpt-oss-20b', 'google/gemini-2.0-flash-001']:
    m = next((x for x in data if x.get('openrouter_id') == mid), None)
    if m:
        cost_per_m = float(m.get('input_cost_per_m', 0))
        cost = cost_per_m / 1000 # Convert /1M to /1k
        print(f"{mid}: Cost=${cost:.6f}/1k, HLE={m.get('hle')}, Description={m.get('description', 'N/A')}")
    else:
        print(f"{mid}: NOT FOUND")
