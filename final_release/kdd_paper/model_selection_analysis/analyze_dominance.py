import json
import os

def load_models(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data['models']

def is_dominated(target, candidate, cluster_idx):
    # Check if candidate dominates target in (Cost, Latency, Success[cluster])
    # Cost: Lower is better
    # Latency: Lower is better
    # Success: Higher is better
    
    # Extract metrics
    t_cost = target.get('price_1m_blended', float('inf'))
    t_latency = target.get('time_to_first_token_seconds', float('inf'))
    t_success = target['cluster_success_rates'][cluster_idx]
    
    c_cost = candidate.get('price_1m_blended', float('inf'))
    c_latency = candidate.get('time_to_first_token_seconds', float('inf'))
    c_success = candidate['cluster_success_rates'][cluster_idx]
    
    # Dominance condition:
    # Candidate must be <= Cost, <= Latency, >= Success
    # AND strictly better in at least one
    
    better_cost = c_cost <= t_cost
    better_latency = c_latency <= t_latency
    better_success = c_success >= t_success
    
    strict = (c_cost < t_cost) or (c_latency < t_latency) or (c_success > t_success)
    
    return better_cost and better_latency and better_success and strict

def find_always_dominated(models):
    always_dominated_models = []
    
    num_clusters = 100 # stated in file, but we can check len
    
    # Verify cluster length for all models
    # Some models might have fewer clusters? The prompt assumes consistent data.
    # We'll assume the length of the first model's cluster list or 100.
    if not models:
        return []
        
    num_clusters = len(models[0]['cluster_success_rates'])
    print(f"Evaluating {len(models)} models across {num_clusters} clusters...")

    for i, target in enumerate(models):
        target_name = target.get('name', target.get('openrouter_id'))
        
        # We need to check if for EVERY cluster, there exists SOME model that dominates 'target'
        dominated_in_all_clusters = True
        
        for cluster_idx in range(num_clusters):
            # Find a dominator for this specific cluster
            found_dominator = False
            for j, candidate in enumerate(models):
                if i == j:
                    continue
                
                if is_dominated(target, candidate, cluster_idx):
                    found_dominator = True
                    # Optional: Print who dominates who in what cluster for debugging
                    # print(f"  Cluster {cluster_idx}: {target_name} dominated by {candidate.get('name', candidate.get('openrouter_id'))}")
                    break
            
            if not found_dominator:
                dominated_in_all_clusters = False
                break # Found a cluster where target is NOT dominated, so it's safer
        
        if dominated_in_all_clusters:
            always_dominated_models.append(target_name)
            
    return always_dominated_models

def main():
    # Look for models.json.bak in likely locations
    possible_paths = [
        'models.json.bak',
        '../../banditgpt/models.json.bak',
        '../../../banditgpt/models.json.bak',
        '/Users/annette/repostitories/llm_jury/banditgpt/models.json.bak'
    ]
    
    path = None
    for p in possible_paths:
        if os.path.exists(p):
            path = p
            break
            
    if not path:
        print("Error: Could not find models.json.bak")
        return

    print(f"Loading from {path}")
    models = load_models(path)
    
    print(f"Total models found: {len(models)}")
    print("Models analyzed:")
    for m in models:
        print(f" - {m.get('name', m.get('openrouter_id'))}")
    
    dominated_names = find_always_dominated(models)
    
    print(f"\nFound {len(dominated_names)} models always dominated on the Cost/Latency/Quality Pareto curve.")
    
    # Calculate stats for non-dominated models for comparison
    non_dominated_throughput = [m.get('output_tokens_per_second', 0) for m in models if m.get('name', m.get('openrouter_id')) not in dominated_names]
    avg_throughput = sum(non_dominated_throughput) / len(non_dominated_throughput) if non_dominated_throughput else 0
    max_throughput = max(non_dominated_throughput) if non_dominated_throughput else 0
    
    print(f"\nBaseline Stats (Non-Dominated Models):")
    print(f"  Avg Throughput: {avg_throughput:.2f} tokens/s")
    print(f"  Max Throughput: {max_throughput:.2f} tokens/s")
    
    print("\nDominated Models Analysis (Throughput Check):")
    print(f"{'Model Name':<50} | {'Throughput':<15} | {'vs Avg':<10}")
    print("-" * 80)
    
    for m in models:
        name = m.get('name', m.get('openrouter_id'))
        if name in dominated_names:
            throughput = m.get('output_tokens_per_second', 0)
            vs_avg = (throughput - avg_throughput) / avg_throughput * 100
            print(f"{name:<50} | {throughput:<15.2f} | {vs_avg:+.1f}%")

if __name__ == "__main__":
    main()
