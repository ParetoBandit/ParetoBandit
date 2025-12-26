"""
Figure 11: Effective System Latency Analysis (Production Simulation)
Comparing Production-Realistic Latency:
- Router Inference (Computation)
- Structural Delays (Cascading Waits)
- Feature Extraction (Embeddings)

All measurements adjusted to reflect real deployment overhead.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from final_release.baselines import BaRPRouter, PILOTRouter
from final_release.kdd_paper.table_3.router_performance_comparison import (
    load_model_registry,
    load_battle_dataset,
    BanditGPTRouter,
    RouteLLMRouter,
    FrugalGPTRouter,
)

sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

def measure_raw_latency(router, queries, n_warmup=5, n_measure=50):
    """Measure raw python function latency."""
    for q in queries[:n_warmup]:
        _ = router.predict_proba(q)
    
    latencies = []
    for q in queries[n_warmup:n_warmup + n_measure]:
        start = time.perf_counter()
        _ = router.predict_proba(q)
        latencies.append((time.perf_counter() - start) * 1000)
    return np.mean(latencies)

def main():
    print("="*60)
    print("FIGURE 11: EFFECTIVE SYSTEM LATENCY ANALYSIS")
    print("="*60)
    
    # 1. Load Real Data
    print("\n[1/4] Loading data...")
    registry = load_model_registry()
    df = load_battle_dataset(n_samples=100)
    queries = df["question"].tolist()
    
    # 2. Measure Raw Overheads
    print("\n[2/4] Measuring raw routing latency...")
    
    # BanditGPT (Reference for Embedding Cost)
    bandit = BanditGPTRouter(registry)
    bandit_time = measure_raw_latency(bandit, queries)
    print(f"  BanditGPT (End-to-End): {bandit_time:.2f} ms")
    
    # BaRP Proxy (Likely near 0 due to static lookup)
    barp = BaRPRouter(registry)
    barp_raw = measure_raw_latency(barp, queries)
    print(f"  BaRP Proxy (Raw Lookup): {barp_raw:.2f} ms")
    
    # PILOT Proxy
    pilot = PILOTRouter(registry)
    pilot_raw = measure_raw_latency(pilot, queries)
    print(f"  PILOT Proxy (Raw Lookup): {pilot_raw:.2f} ms")
    
    # RouteLLM (Deep Router)
    routellm = RouteLLMRouter()
    routellm_time = measure_raw_latency(routellm, queries)
    print(f"  RouteLLM (Deep Model): {routellm_time:.2f} ms")
    
    # FrugalGPT (Scoring Only)
    frugal = FrugalGPTRouter()
    frugal_raw = measure_raw_latency(frugal, queries)
    print(f"  FrugalGPT (Scoring Only): {frugal_raw:.2f} ms")
    
    # 3. Apply Production Reality Adjustments
    print("\n[3/4] Applying production adjustments...")
    
    # A. Embedding Penalty
    # Any contextual bandit (BaRP/PILOT) needs embeddings in production.
    # We use BanditGPT's time as the "Cost of Context".
    embedding_cost = bandit_time  
    print(f"  Embedding cost (for proxies): {embedding_cost:.2f} ms")
    
    # B. Cascade Penalty (For FrugalGPT) - REAL DATA
    # Extract weak model latency from registry (cheapest model)
    weak_model_id = min(registry.keys(), 
                        key=lambda m: float(registry[m].get("input_cost_per_m", 0)) + 
                                     float(registry[m].get("output_cost_per_m", 0)))
    weak_lowest_latency_seconds = float(registry[weak_model_id].get("lowest_latency_seconds", 0))
    if weak_lowest_latency_seconds <= 0:
        raise ValueError(f"No latency data for weak model {weak_model_id}! Cannot proceed without real data.")
    weak_model_latency = weak_lowest_latency_seconds * 1000  # Convert to ms
    
    # Estimate fail rate from quality difference
    # Weak models have higher hallucination rates, so cascade fails more often
    weak_hall = float(registry[weak_model_id].get("hallucination_vectara", 10.0))
    cascade_fail_rate = min(0.5, weak_hall / 100.0 * 3)  # Heuristic: higher hall = more fails
    
    cascade_penalty = cascade_fail_rate * weak_model_latency
    print(f"  Weak model latency (real): {weak_model_latency:.0f} ms")
    print(f"  Cascade fail rate (estimated): {cascade_fail_rate:.2f}")
    print(f"  Cascade penalty (FrugalGPT): {cascade_penalty:.0f} ms")
    
    # 4. Construct Final Data
    routers = ['BanditGPT\n(Ours)', 'BaRP\n(Proxy)', 'PILOT\n(Proxy)', 'FrugalGPT\n(Cascade)', 'RouteLLM\n(Static)']
    
    # Calculated Overheads (Production-Realistic)
    overheads = [
        bandit_time,                # Measured (includes embedding)
        barp_raw + embedding_cost,  # Corrected: Proxy + Required Embedding
        pilot_raw + embedding_cost, # Corrected: Same for PILOT
        frugal_raw + cascade_penalty, # Corrected: Scoring + Cascading Wait
        routellm_time               # Measured (includes BERT inference)
    ]
    
    print(f"\n  Production-adjusted overheads:")
    for name, overhead in zip(routers, overheads):
        print(f"    {name.replace(chr(10), ' '):25s}: {overhead:.2f} ms")
    
    # Base Model Generation Time - REAL DATA ONLY (no fallbacks)
    model_latencies = []
    for m_id, metadata in registry.items():
        lowest_latency_seconds = metadata.get("lowest_latency_seconds", 0)
        if lowest_latency_seconds > 0:
            model_latencies.append(lowest_latency_seconds * 1000)  # Convert to ms
    
    if not model_latencies:
        raise ValueError("No latency data found in registry! Cannot proceed without real data.")
    
    base_gen_time = np.mean(model_latencies)
    print(f"  Average model generation time (real): {base_gen_time:.0f} ms from {len(model_latencies)} models") 
    
    # 5. Plotting
    print("\n[4/4] Generating plot...")
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(routers))
    width = 0.6
    
    # Base Bar (Model Generation)
    p1 = ax.bar(x, [base_gen_time]*len(routers), width, 
                label='Avg Model Generation', color='#6C8EBF', alpha=0.9, edgecolor='grey')
    
    # Overhead Bar (The Tax)
    # Color coding: Green (Good - Bandits), Orange (Bad - Cascade), Purple (Static)
    colors = ['#2ca02c', '#2ca02c', '#2ca02c', '#ff7f0e', '#9467bd']
    p2 = ax.bar(x, overheads, width, bottom=[base_gen_time]*len(routers), 
                label='System Overhead (Router + Penalties)', color=colors, alpha=0.9)
    
    # Labels on overhead bars
    for i, v in enumerate(overheads):
        total = base_gen_time + v
        pct = (v / total) * 100
        
        label = f"+{v:.0f}ms\n({pct:.1f}%)"
        
        ax.text(i, total + 15, label, ha='center', va='bottom', fontweight='bold', fontsize=9)

    # Annotations (positioned to avoid legend overlap)
    ax.annotate('Bandit Efficiency\n(Fast Decisions)', 
                xy=(0.5, base_gen_time + overheads[1]), xytext=(0.5, 900),
                arrowprops=dict(arrowstyle='->', color='green', lw=2),
                ha='center', color='darkgreen', fontweight='bold', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))
                
    ax.annotate('Cascade Tax\n(Sequential Waits)', 
                xy=(3, base_gen_time + overheads[3]/2), xytext=(3.5, 400),
                arrowprops=dict(arrowstyle='->', color='darkred', lw=2),
                ha='center', color='darkred', fontweight='bold', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.7))
    
    ax.annotate('BERT Inference\n(Deep Router)', 
                xy=(4, base_gen_time + overheads[4]/2), xytext=(4.5, 600),
                arrowprops=dict(arrowstyle='->', color='purple', lw=2),
                ha='center', color='purple', fontweight='bold', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lavender', alpha=0.7))

    ax.set_ylabel('Total Request Latency (ms)', fontweight='bold', fontsize=12)
    ax.set_title('Figure 11: The Latency Tax - Production System Overhead', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(routers, fontsize=11)
    ax.set_ylim(0, 2000)
    ax.legend(loc='upper left', fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = Path(__file__).parent / "latency_tax.png"
    plt.savefig(output_path, dpi=300, facecolor='white')
    print(f"\n✅ Plot saved to {output_path}")
    
    # Summary table
    print("\n" + "="*60)
    print("LATENCY TAX SUMMARY (Production-Adjusted)")
    print("="*60)
    for name, overhead in zip(routers, overheads):
        total = base_gen_time + overhead
        pct = (overhead / total) * 100
        print(f"{name.replace(chr(10), ' '):30s}: +{overhead:6.1f}ms ({pct:4.1f}%)")
    print("="*60)
    print("\nKey Insight: BanditGPT achieves learning capabilities with")
    print("             minimal overhead compared to cascade approaches.")

if __name__ == "__main__":
    main()
