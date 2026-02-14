#!/usr/bin/env python3
"""
Figure 4: Cost-Aware Model Discovery — GPT-4o Introduced Mid-Stream

Uses the PRODUCTION router classes (CostAwareLinUCBRouter, CostAwareTabulaRasaRouter)
with cost-aware UCB scoring. Runs a cost_penalty ablation (lambda = 0.0, 0.1, 0.3, 0.5)
to show how cost sensitivity changes routing behavior when a new model is added.

The experiment has two phases on a single timeline:
  Phase 1: 2-model portfolio (Mixtral + GPT-4-Turbo)
  Phase 2: GPT-4o added via semantic transfer → 3-model portfolio

The plot shows rolling model selection frequencies, with a vertical line marking
GPT-4o's introduction. Different lambda values produce different cost-quality tradeoffs.
"""

import sys
import math
from pathlib import Path

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import json
import copy
import gzip
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from bandit_gpt.calibration import apply_gamma_scaling, embed_prompt
from bandit_gpt.router import (
    CorrallingRouter,
    CostAwareLinUCBRouter,
    CostAwareTabulaRasaRouter,
)
from bandit_gpt.config_legacy import (
    DEFAULT_SENTENCE_TRANSFORMER,
    DEFAULT_WARMUP_PRIORS_PATH,
    DEFAULT_PCA_PATH,
    OFFLINE_DATASET_DIR,
)

# Import helper functions from main script
sys.path.insert(0, str(Path(__file__).parent))
from corralled_semantic_analysis import (
    load_labeled_data,
    compute_oracle_reward,
    extend_priors_with_semantic_transfer,
)

CANONICAL_DEV_DATA_PATH = OFFLINE_DATASET_DIR / "dev_rewards_complete.jsonl.gz"


# ============================================================================
# Model cost infrastructure (matches production BanditRouter normalization)
# ============================================================================

# Market anchors from RouterConfig
MARKET_COST_FLOOR = 0.0001   # $/1k tokens
MARKET_COST_CEILING = 0.04   # $/1k tokens

# Raw costs from models.json / models_all.json ($/1M tokens)
MODEL_COSTS_RAW = {
    'mistralai/mixtral-8x7b-instruct': {'input_cost_per_m': 0.54, 'output_cost_per_m': 0.60},
    'openai/gpt-4-turbo':              {'input_cost_per_m': 10.0, 'output_cost_per_m': 30.0},
    'openai/gpt-4o':                   {'input_cost_per_m': 2.5,  'output_cost_per_m': 10.0},
}


def compute_normalized_cost(input_cost_per_m: float, output_cost_per_m: float) -> float:
    """
    Normalize cost to [0, 1] using log-scale market anchors.
    Matches production _calculate_absolute_penalty() in router.py.
    """
    avg_cost_per_1k = ((input_cost_per_m + output_cost_per_m) / 2.0) / 1000.0
    safe_cost = max(avg_cost_per_1k, MARKET_COST_FLOOR)
    log_cost = math.log(safe_cost)
    log_floor = math.log(MARKET_COST_FLOOR)
    log_range = math.log(MARKET_COST_CEILING) - log_floor
    return max(0.0, min(1.0, (log_cost - log_floor) / log_range))


def build_model_costs(model_ids: list) -> dict:
    """Build model_costs dict in the format CostAwareLinUCBRouter expects."""
    model_costs = {}
    for m in model_ids:
        raw = MODEL_COSTS_RAW.get(m, {'input_cost_per_m': 10.0, 'output_cost_per_m': 30.0})
        norm = compute_normalized_cost(raw['input_cost_per_m'], raw['output_cost_per_m'])
        model_costs[m] = {'normalized_cost': norm}
    return model_costs


# ============================================================================
# Core experiment
# ============================================================================

def run_corralling(labeled_data, encoder, pca, warmup_priors_raw, model_subset,
                   cost_penalty=0.0, label="", enable_transfer=True, seed=None):
    """
    Run Corralling on a subset of models using production CostAware* classes.

    Args:
        labeled_data: List of samples with 'scores' dict
        encoder: SentenceTransformer
        pca: PCA model
        warmup_priors_raw: UNSCALED warmup priors (will be extended + scaled internally)
        model_subset: List of model IDs to include
        cost_penalty: Lambda for cost-aware UCB (0.0 = quality-only)
        label: Display label for progress bar
        enable_transfer: Whether to use semantic transfer for missing models
        seed: Random seed for reproducibility. If None, uses global RNG state.
    """
    if seed is not None:
        np.random.seed(seed)
    priors = copy.deepcopy(warmup_priors_raw)

    # Extend priors for models not in warmup (e.g. GPT-4o)
    missing = [m for m in model_subset if m not in priors['models']]
    if missing and enable_transfer:
        transfer_mapping = {'openai/gpt-4o': 'openai/gpt-4-turbo'}
        priors = extend_priors_with_semantic_transfer(
            priors, new_models=missing,
            transfer_mapping=transfer_mapping,
        )

    # Filter priors to only include models in model_subset
    priors_filtered = {
        'A': {m: priors['A'][m] for m in model_subset if m in priors['A']},
        'b': {m: priors['b'][m] for m in model_subset if m in priors['b']},
        'models': [m for m in model_subset if m in priors['A']],
        'context_dim': priors['context_dim']
    }
    priors_scaled = apply_gamma_scaling(priors_filtered, gamma=0.05)

    models = priors_scaled['models']
    context_dim = priors_scaled['A'][models[0]].shape[0]
    model_costs = build_model_costs(models)

    # Production router classes
    # NOTE: cost_penalty=0 on experts — cost is handled via composite reward
    # in CorrallingRouter (cost_weight) to avoid double-counting.
    warmup_expert = CostAwareLinUCBRouter(
        models=models,
        warmup_priors=priors_scaled,
        model_costs=model_costs,
        alpha_start=2.0,
        alpha_end=0.1,
        cost_penalty=0.0,
    )
    tabula_rasa_expert = CostAwareTabulaRasaRouter(
        models=models,
        context_dim=context_dim,
        model_costs=model_costs,
        alpha_start=1.0,
        alpha_end=0.01,
        cost_penalty=0.0,
        ridge_lambda=1.0,
    )
    router = CorrallingRouter(
        experts=[warmup_expert, tabula_rasa_expert],
        models=models,
        learning_rate=1.0,
        model_costs=model_costs,
        cost_weight=cost_penalty,  # composite reward: r = quality - λ·cost
    )

    per_step_selections = []
    per_step_rewards = []
    cumulative_regret = 0.0
    total_reward = 0.0
    total_steps = len(labeled_data)

    for i, sample in enumerate(tqdm(labeled_data, desc=f"   {label}")):
        context = embed_prompt(sample['prompt'], encoder, pca)
        selected_model, selection_token = router.select_model(
            context, total_steps=total_steps)

        # Restrict oracle to models in this subset
        subset_scores = {m: sample['scores'][m] for m in model_subset
                         if m in sample['scores']}
        model_reward = subset_scores.get(selected_model, 0.0)
        oracle_reward = max(subset_scores.values()) if subset_scores else 0.0

        cumulative_regret += (oracle_reward - model_reward)
        total_reward += model_reward
        per_step_selections.append(selected_model)
        per_step_rewards.append(model_reward)

        router.update(context, selected_model, model_reward,
                      selection_token=selection_token)

    return {
        'models': models,
        'per_step_selections': per_step_selections,
        'per_step_rewards': per_step_rewards,
        'cumulative_regret': cumulative_regret,
        'avg_reward': total_reward / len(labeled_data),
        'final_weights': router.weights.tolist(),
        'model_usage': dict(router.selections),
    }


# ============================================================================
# Main: ablation + multi-panel figure
# ============================================================================

def main():
    print("=" * 80)
    print("FIGURE 4: COST-AWARE MODEL DISCOVERY — ABLATION")
    print("=" * 80)

    # Load resources
    print("\n Loading resources...")
    encoder = SentenceTransformer(DEFAULT_SENTENCE_TRANSFORMER)
    pca = joblib.load(DEFAULT_PCA_PATH)
    warmup_priors = joblib.load(DEFAULT_WARMUP_PRIORS_PATH)

    # Load data
    print("\n Loading labeled data...")
    labeled_data = load_labeled_data(Path(CANONICAL_DEV_DATA_PATH), sample_size=1121)
    print(f"   Loaded {len(labeled_data)} samples")

    two_models = ['mistralai/mixtral-8x7b-instruct', 'openai/gpt-4-turbo']
    three_models = ['mistralai/mixtral-8x7b-instruct', 'openai/gpt-4-turbo', 'openai/gpt-4o']

    # Print normalized costs
    mc = build_model_costs(three_models)
    print("\n   Normalized costs:")
    for m in three_models:
        raw = MODEL_COSTS_RAW[m]
        print(f"      {m:<40} ${raw['input_cost_per_m']}/{raw['output_cost_per_m']}/M "
              f"→ norm={mc[m]['normalized_cost']:.3f}")

    # Split at midpoint
    switch_point = len(labeled_data) // 2
    phase1_data = labeled_data[:switch_point]
    phase2_data = labeled_data[switch_point:]

    print(f"\n   Phase 1: samples 1–{switch_point} (2 models)")
    print(f"   Phase 2: samples {switch_point + 1}–{len(labeled_data)} "
          f"(3 models, GPT-4o added)")

    # ================================================================
    # Run ablation over cost_penalty values (multi-seed for reliability)
    # ================================================================
    lambda_values = [0.0, 0.1, 0.3, 0.5]
    N_SEEDS = 5
    seeds = list(range(42, 42 + N_SEEDS))
    PLOT_SEED = 42                  # Same seed for all panels → λ is the only variable
    ablation_results = {}           # lam -> fixed-seed result (for plot)
    ablation_multi_seed = {}        # lam -> list of per-seed results

    for lam in lambda_values:
        print(f"\n{'=' * 60}")
        print(f"  COST PENALTY λ = {lam}  ({N_SEEDS} seeds)")
        print(f"{'=' * 60}")

        seed_results = []
        for s_idx, seed in enumerate(seeds):
            # Phase 1: 2-model router — same λ as Phase 2
            # λ is applied consistently across both phases so the ONLY
            # change at the dashed line is GPT-4o being added (no confound).
            result_p1 = run_corralling(
                phase1_data, encoder, pca, warmup_priors,
                model_subset=two_models, cost_penalty=lam,
                label=f"λ={lam} seed={seed} P1", enable_transfer=False,
                seed=seed)

            # Phase 2: 3-model router (GPT-4o added, same λ continues)
            result_p2 = run_corralling(
                phase2_data, encoder, pca, warmup_priors,
                model_subset=three_models, cost_penalty=lam,
                label=f"λ={lam} seed={seed} P2", enable_transfer=True,
                seed=seed + 1000)  # Different seed for phase 2

            # Splice into single timeline
            all_sel = result_p1['per_step_selections'] + result_p2['per_step_selections']
            all_rew = result_p1['per_step_rewards'] + result_p2['per_step_rewards']

            seed_results.append({
                'seed': seed,
                'phase1': result_p1,
                'phase2': result_p2,
                'all_selections': all_sel,
                'all_rewards': all_rew,
                'avg_reward': float(np.mean(all_rew)),
            })

        ablation_multi_seed[lam] = seed_results

        # Use the FIXED seed (seed=42) for plot — same seed across all λ
        # so the ONLY variable between panels is λ (controlled comparison)
        plot_idx = next(i for i, r in enumerate(seed_results) if r['seed'] == PLOT_SEED)
        ablation_results[lam] = seed_results[plot_idx]

        # Summary (averaged over seeds)
        avg_rewards = [r['avg_reward'] for r in seed_results]
        print(f"\n   Results over {N_SEEDS} seeds:")
        print(f"   Avg reward: {np.mean(avg_rewards):.4f} ± {np.std(avg_rewards):.4f}")

        # Aggregate Phase 2 usage percentages across seeds
        _short = {
            'openai/gpt-4o': 'GPT-4o',
            'openai/gpt-4-turbo': 'GPT-4-Turbo',
            'mistralai/mixtral-8x7b-instruct': 'Mixtral',
        }
        for m in three_models:
            pcts = []
            for r in seed_results:
                p2u = r['phase2']['model_usage']
                p2t = sum(p2u.values())
                pcts.append(100 * p2u.get(m, 0) / p2t)
            print(f"      {_short[m]:<15} {np.mean(pcts):>5.1f}% ± {np.std(pcts):>4.1f}%")

    # ================================================================
    # Figure: 2x2 panel ablation
    # ================================================================

    output_dir = Path(__file__).parent / "results_3models"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    T = len(labeled_data)
    window = 50

    colors = {
        'openai/gpt-4o': '#1f77b4',
        'openai/gpt-4-turbo': '#d62728',
        'mistralai/mixtral-8x7b-instruct': '#2ca02c',
    }
    short = {
        'openai/gpt-4o': 'GPT-4o',
        'openai/gpt-4-turbo': 'GPT-4-Turbo',
        'mistralai/mixtral-8x7b-instruct': 'Mixtral',
    }

    # Human-readable regime labels for each lambda
    regime_labels = {
        0.0: 'Quality Only',
        0.1: 'Mild Cost Awareness',
        0.3: 'Moderate Cost Awareness',
        0.5: 'Aggressive Cost Preference',
    }

    # Average cost per query ($/1M tokens, blended input+output)
    cost_per_query = {
        'mistralai/mixtral-8x7b-instruct': (0.54 + 0.60) / 2,
        'openai/gpt-4-turbo': (10.0 + 30.0) / 2,
        'openai/gpt-4o': (2.5 + 10.0) / 2,
    }
    gpt4o_cost = cost_per_query['openai/gpt-4o']

    # Pre-compute Phase 2 metrics AVERAGED OVER ALL SEEDS (for summary boxes)
    panel_metrics = {}
    for lam in lambda_values:
        all_qualities = []
        all_costs = []
        for r in ablation_multi_seed[lam]:
            p2_sel = r['phase2']['per_step_selections']
            p2_rew = r['phase2']['per_step_rewards']
            all_qualities.append(np.mean(p2_rew))
            all_costs.append(np.mean([cost_per_query.get(s, 6.25) for s in p2_sel]))
        mean_quality = np.mean(all_qualities)
        std_quality = np.std(all_qualities)
        mean_cost = np.mean(all_costs)
        std_cost = np.std(all_costs)
        cost_savings = (1 - mean_cost / gpt4o_cost) * 100 if mean_cost < gpt4o_cost else 0
        panel_metrics[lam] = {
            'quality': mean_quality,
            'quality_std': std_quality,
            'avg_cost': mean_cost,
            'cost_std': std_cost,
            'savings': cost_savings,
        }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True, sharex=True)
    axes_flat = axes.flatten()

    for idx, lam in enumerate(lambda_values):
        ax = axes_flat[idx]
        res = ablation_results[lam]
        all_sel = res['all_selections']
        pm = panel_metrics[lam]

        # Rolling fractions
        steps = np.arange(1, T + 1)
        indicators = {m: np.array([1.0 if s == m else 0.0 for s in all_sel])
                      for m in three_models}
        kernel = np.ones(window) / window
        rolling = {m: np.convolve(indicators[m], kernel, mode='valid')
                   for m in three_models}
        r_steps = steps[window - 1:]

        # Plot lines
        for m in ['openai/gpt-4o', 'openai/gpt-4-turbo', 'mistralai/mixtral-8x7b-instruct']:
            ax.plot(r_steps, rolling[m] * 100, linewidth=2.0,
                    color=colors[m], label=short[m])

        # Vertical switch line
        ax.axvline(x=switch_point, linestyle='--', linewidth=1.5,
                   color='black', alpha=0.5, zorder=4)

        # Shade phases
        ax.axvspan(r_steps[0], switch_point, alpha=0.03, color='gray')
        ax.axvspan(switch_point, r_steps[-1], alpha=0.03, color='#1f77b4')

        # "GPT-4o added" label — small, at the top of the dashed line,
        # inside the plot but clear of data (data rarely reaches 100%)
        ax.text(switch_point + 5, 101, 'GPT-4o added', fontsize=7,
                ha='left', va='bottom', color='#444444', fontstyle='italic')

        # ---- Summary box (Phase 2 outcomes, averaged over seeds) ----
        savings_line = (f"\nSavings: {pm['savings']:.0f}% vs GPT-4o"
                        if pm['savings'] > 1 else "")
        summary_text = (
            f"Phase 2 ({N_SEEDS}-seed avg):\n"
            f"  Quality  {pm['quality']:.1%} ± {pm['quality_std']:.1%}\n"
            f"  Avg cost ${pm['avg_cost']:.2f} ± ${pm['cost_std']:.2f}/M"
            f"{savings_line}"
        )
        ax.text(0.97, 0.97, summary_text,
                transform=ax.transAxes, fontsize=7.5, va='top', ha='right',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                         edgecolor='#999999', alpha=0.95),
                family='monospace', linespacing=1.3, zorder=10)

        # ---- Panel title: regime + lambda + key metric ----
        regime = regime_labels.get(lam, '')
        title = f'{regime}  ($\\lambda$ = {lam})'
        ax.set_title(title, fontsize=11, fontweight='bold', pad=10)

        # Axis labels
        if idx >= 2:
            ax.set_xlabel('Training Samples', fontsize=11, fontweight='bold')
        if idx % 2 == 0:
            ax.set_ylabel('Selection Frequency (%)\n(rolling window = 50)',
                         fontsize=9.5, fontweight='bold')

        ax.set_ylim([-2, 105])
        ax.set_xlim([r_steps[0], r_steps[-1]])
        ax.grid(True, alpha=0.15, linestyle='--')

        # Legend only on first panel
        if idx == 0:
            ax.legend(loc='upper left', fontsize=8.5, framealpha=0.95,
                      edgecolor='#cccccc')

    # ---- Suptitle ----
    fig.suptitle(
        'Cost-Aware Routing: GPT-4o Introduced Mid-Stream\n'
        'Composite reward $r = \\mathrm{quality} - \\lambda \\cdot \\mathrm{cost}$'
        f'  (same seed={PLOT_SEED}; summary boxes show {N_SEEDS}-seed avg)',
        fontsize=13, fontweight='bold', y=1.03, linespacing=1.4)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_dir / 'figure4_model_discovery.png', dpi=300,
                bbox_inches='tight', facecolor='white')
    plt.savefig(results_dir / 'figure4_model_discovery.png', dpi=300,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n Saved figure: {results_dir / 'figure4_model_discovery.png'}")

    # ================================================================
    # Save results JSON
    # ================================================================

    from collections import defaultdict
    model_rewards_all = defaultdict(list)
    for sample in labeled_data:
        for m, s in sample['scores'].items():
            model_rewards_all[m].append(s)
    model_mean_rewards = {m: float(np.mean(v)) for m, v in model_rewards_all.items()}

    results = {
        'n_models': 3,
        'models': three_models,
        'train_size': len(labeled_data),
        'switch_point': switch_point,
        'model_mean_rewards': model_mean_rewards,
        'normalized_costs': {m: mc[m]['normalized_cost'] for m in three_models},
        'ablation': {},
    }

    for lam in lambda_values:
        res = ablation_results[lam]
        results['ablation'][str(lam)] = {
            'cost_penalty': lam,
            'phase1_usage': res['phase1']['model_usage'],
            'phase2_usage': res['phase2']['model_usage'],
            'avg_reward': res['avg_reward'],
            'per_step_selections': res['all_selections'],
            'per_step_rewards': res['all_rewards'],
        }

    with open(output_dir / 'quick_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n Saved results to: {output_dir}/quick_test_results.json")

    # ================================================================
    # Summary table (multi-seed averaged)
    # ================================================================
    print(f"\n{'=' * 90}")
    print(f"ABLATION SUMMARY ({N_SEEDS}-seed averages)")
    print(f"{'=' * 90}")
    print(f"\n{'Lambda':<10} {'Quality':<16} {'Avg Cost ($/M)':<20} "
          f"{'Mixtral %':<14} {'GPT-4-Turbo %':<16} {'GPT-4o %':<12}")
    print("-" * 90)
    for lam in lambda_values:
        pm = panel_metrics[lam]
        # Aggregate Phase 2 usage across seeds
        mix_pcts, turbo_pcts, gpt4o_pcts = [], [], []
        for r in ablation_multi_seed[lam]:
            p2u = r['phase2']['model_usage']
            p2t = sum(p2u.values())
            mix_pcts.append(100 * p2u.get('mistralai/mixtral-8x7b-instruct', 0) / p2t)
            turbo_pcts.append(100 * p2u.get('openai/gpt-4-turbo', 0) / p2t)
            gpt4o_pcts.append(100 * p2u.get('openai/gpt-4o', 0) / p2t)
        print(f"{lam:<10.1f} "
              f"{pm['quality']:.1%} ± {pm['quality_std']:.1%}   "
              f"${pm['avg_cost']:>5.2f} ± ${pm['cost_std']:.2f}     "
              f"{np.mean(mix_pcts):>5.1f}±{np.std(mix_pcts):.1f}    "
              f"{np.mean(turbo_pcts):>5.1f}±{np.std(turbo_pcts):.1f}      "
              f"{np.mean(gpt4o_pcts):>5.1f}±{np.std(gpt4o_pcts):.1f}")

    print(f"\n{'=' * 80}")
    print(" EXPERIMENT COMPLETE!")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    main()
