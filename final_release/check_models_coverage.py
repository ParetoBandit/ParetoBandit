import json
from pathlib import Path

def main():
    base_dir = Path(__file__).parent
    models_path = base_dir / "models.json"
    
    with open(models_path) as f:
        data = json.load(f)
        
    models = data["models"]
    total = len(models)
    
    has_hle = 0
    has_cost = 0
    has_latency = 0
    
    print(f"Checking {total} models in {models_path}...")
    
    missing_hle = []
    missing_cost = []
    missing_latency = []
    
    print(f"Checking {total} models in {models_path}...")
    
    for m in models:
        name = m.get("openrouter_id", m.get("name"))
        
        # Check HLE or reasoning_score (fallback)
        hle = m.get("hle")
        reasoning = m.get("reasoning_score")
        if (hle is None or hle == 0) and (reasoning is None or reasoning == 0):
            missing_hle.append(name)
        else:
            has_hle += 1
            
        # Check Cost
        in_cost = m.get("input_cost_per_m")
        out_cost = m.get("output_cost_per_m")
        if in_cost is None or out_cost is None or (in_cost == 0 and out_cost == 0):
            missing_cost.append(name)
        else:
            has_cost += 1
            
        # Check Latency
        ttft = m.get("time_to_first_token_seconds")
        otps = m.get("output_tokens_per_second")
        if ttft is None or otps is None or (ttft == 0 and otps == 0):
            missing_latency.append(name)
        else:
            has_latency += 1
            
    print(f"HLE Coverage:     {has_hle}/{total} ({has_hle/total*100:.1f}%)")
    print(f"Cost Coverage:    {has_cost}/{total} ({has_cost/total*100:.1f}%)")
    print(f"Latency Coverage: {has_latency}/{total} ({has_latency/total*100:.1f}%)")
    
    if missing_hle:
        print(f"\nMissing HLE ({len(missing_hle)}): {missing_hle}")
    if missing_cost:
        print(f"\nMissing Cost ({len(missing_cost)}): {missing_cost}")
    if missing_latency:
        print(f"\nMissing Latency ({len(missing_latency)}): {missing_latency}")

if __name__ == "__main__":
    main()
