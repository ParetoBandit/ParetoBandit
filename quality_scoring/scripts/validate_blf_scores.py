#!/usr/bin/env python3
"""
Validate Bayesian Latent Factor (BLF) Composite Scores

This script performs quality checks on the data used for each composite score:
- CCS (Composite Coding Score)
- CRS (Composite Reasoning Score)  
- CFS (Composite Factual QA Score)
- CSS (Composite Summarization Score)

Checks performed:
1. Connectivity: Ensures all models are in a single connected component
2. Anchor Rule: Verifies each model has at least one high-quality anchor metric
3. Sparsity: Reports matrix sparsity and warns if too sparse
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass

import pandas as pd
import networkx as nx

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@dataclass
class CompositeScoreConfig:
    """Configuration for a composite score validation."""
    name: str
    description: str
    benchmarks: List[str]  # Primary benchmarks
    anchors: Set[str]  # Required anchor metrics
    score_field: str
    auxiliary: List[str] = None  # Auxiliary benchmarks for covariance imputation
    
    def __post_init__(self):
        if self.auxiliary is None:
            self.auxiliary = []
    
    @property
    def all_benchmarks(self) -> List[str]:
        """Get all benchmarks (primary + auxiliary)."""
        return self.benchmarks + self.auxiliary


# Define all composite score configurations
COMPOSITE_SCORES = {
    'CRS': CompositeScoreConfig(
        name='CRS',
        description='Composite Reasoning Score',
        benchmarks=['math_500', 'gpqa', 'hle', 'aime', 'math_index'],
        anchors={'gpqa', 'math_500'},  # Most discriminating & widely available
        score_field='reasoning_score',
    ),
    'CCS': CompositeScoreConfig(
        name='CCS',
        description='Composite Coding Score',
        benchmarks=['humaneval_score', 'livecodebench', 'scicode', 'arena_rank_coding'],
        anchors={'humaneval_score', 'arena_rank_coding'},  # Most reliable & widely available
        score_field='ccs_100',
        auxiliary=['intelligence_index'],  # r=0.96 with coding, 100% coverage
    ),
    'CFS': CompositeScoreConfig(
        name='CFS',
        description='Composite Factual QA Score',
        benchmarks=['mmlu_pro', 'gpqa', 'arena_rank_expert'],
        anchors={'mmlu_pro', 'gpqa'},  # Most established & discriminating
        score_field='cfs_100',
    ),
    'CSS': CompositeScoreConfig(
        name='CSS', 
        description='Composite Summarization Score',
        benchmarks=['summedits_score', 'hallucination_rate', 'arena_rank_longer'],
        anchors={'summedits_score', 'hallucination_rate'},  # Primary quality signals
        score_field='css_100',
    ),
}


def load_models_cache() -> List[Dict]:
    """Load models from cache file."""
    cache_path = Path(__file__).parent.parent.parent / "data" / "models_cache.json"
    with open(cache_path, 'r') as f:
        data = json.load(f)
    return data.get('models', data) if isinstance(data, dict) else data


def build_bipartite_graph(
    models: List[Dict], 
    benchmarks: List[str]
) -> Tuple[nx.Graph, List[str], List[str]]:
    """Build bipartite graph of models <-> benchmarks."""
    B = nx.Graph()
    
    model_names = []
    benchmark_set = set(benchmarks)
    edges = []
    
    for m in models:
        name = m.get('name', '')
        if not name:
            continue
        model_names.append(name)
        
        for bench in benchmarks:
            val = m.get(bench)
            if val is not None:
                edges.append((name, bench))
    
    # Add nodes with bipartite attribute
    B.add_nodes_from(model_names, bipartite=0, type='model')
    B.add_nodes_from(benchmarks, bipartite=1, type='benchmark')
    B.add_edges_from(edges)
    
    return B, model_names, benchmarks


def validate_composite_score(
    config: CompositeScoreConfig,
    models: List[Dict],
    verbose: bool = True
) -> Dict:
    """
    Validate a single composite score.
    
    Returns dict with validation results.
    """
    results = {
        'name': config.name,
        'description': config.description,
        'passed': True,
        'warnings': [],
        'errors': [],
    }
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"  {config.name}: {config.description}")
        print(f"{'='*70}")
        print(f"  Primary Benchmarks: {', '.join(config.benchmarks)}")
        if config.auxiliary:
            print(f"  Auxiliary Benchmarks: {', '.join(config.auxiliary)} (for covariance imputation)")
        print(f"  Anchors: {', '.join(config.anchors)}")
        print()
    
    # Build the graph with ALL benchmarks (primary + auxiliary)
    B, model_names, benchmarks = build_bipartite_graph(models, config.all_benchmarks)
    
    # Filter to models that have at least one benchmark
    models_with_data = [m for m in model_names if B.degree(m) > 0]
    
    if verbose:
        print(f"  Models with data: {len(models_with_data)} / {len(model_names)}")
    
    # ==========================================
    # CHECK 1: Connectivity (Golden Rule)
    # ==========================================
    # Only check connectivity among models that have data
    subgraph = B.subgraph(models_with_data + config.benchmarks)
    components = list(nx.connected_components(subgraph))
    
    if verbose:
        print(f"\n--- CHECK 1: Connectivity ---")
    
    if len(components) == 1:
        if verbose:
            print(f"✅ PASSED: Graph is fully connected ({len(models_with_data)} models, {len(config.benchmarks)} benchmarks)")
        results['connectivity'] = 'passed'
        results['num_components'] = 1
    else:
        if verbose:
            print(f"❌ FAILED: Found {len(components)} disconnected islands!")
        results['passed'] = False
        results['connectivity'] = 'failed'
        results['num_components'] = len(components)
        results['errors'].append(f"Found {len(components)} disconnected components")
        
        # Show details of each island
        for i, comp in enumerate(components):
            comp_models = [n for n in comp if n in models_with_data]
            comp_benchmarks = [n for n in comp if n in config.benchmarks]
            if verbose:
                print(f"   Island {i+1}: {len(comp_models)} models, benchmarks: {comp_benchmarks}")
                if len(comp_models) <= 5:
                    print(f"            Models: {comp_models}")
    
    # ==========================================
    # CHECK 2: Anchor Rule
    # ==========================================
    if verbose:
        print(f"\n--- CHECK 2: Anchor Rule ---")
    
    failed_anchors = []
    models_with_aux_only = []  # Models rescued by auxiliary benchmarks
    anchor_coverage = {a: 0 for a in config.anchors}
    
    for m in models_with_data:
        my_benchmarks = set(B.neighbors(m))
        intersection = my_benchmarks.intersection(config.anchors)
        
        # Count anchor coverage
        for a in intersection:
            anchor_coverage[a] += 1
        
        if not intersection:
            # Check if they have auxiliary benchmarks
            if config.auxiliary and my_benchmarks.intersection(set(config.auxiliary)):
                models_with_aux_only.append(m)
            else:
                failed_anchors.append(m)
    
    # Report anchor coverage
    if verbose:
        print(f"  Anchor coverage:")
        for anchor, count in sorted(anchor_coverage.items(), key=lambda x: -x[1]):
            pct = 100 * count / len(models_with_data) if models_with_data else 0
            print(f"    {anchor}: {count}/{len(models_with_data)} ({pct:.0f}%)")
    
    results['anchor_coverage'] = anchor_coverage
    
    # Report auxiliary rescue
    if config.auxiliary and models_with_aux_only:
        if verbose:
            print(f"\n  Covariance Imputation (via auxiliary benchmarks):")
            print(f"    {len(models_with_aux_only)} models lack primary anchors but have auxiliary coverage")
            print(f"    -> These models will borrow strength from correlated auxiliary metrics")
        results['models_rescued_by_auxiliary'] = models_with_aux_only
    
    if not failed_anchors and not models_with_aux_only:
        if verbose:
            print(f"✅ PASSED: Every model has at least 1 anchor metric")
        results['anchor_rule'] = 'passed'
        results['models_without_anchor'] = []
    elif not failed_anchors:
        if verbose:
            print(f"✅ PASSED: All models anchored (some via auxiliary covariance)")
        results['anchor_rule'] = 'passed_with_auxiliary'
        results['models_without_anchor'] = []
    else:
        if verbose:
            print(f"⚠️  WARNING: {len(failed_anchors)} models lack ANY anchor (primary or auxiliary):")
        results['anchor_rule'] = 'warning'
        results['warnings'].append(f"{len(failed_anchors)} models lack anchor metrics")
        results['models_without_anchor'] = failed_anchors
        
        if verbose and len(failed_anchors) <= 10:
            for m in failed_anchors:
                benchs = list(B.neighbors(m))
                print(f"      - {m}: has {benchs}")
        elif verbose:
            print(f"      (showing first 10)")
            for m in failed_anchors[:10]:
                benchs = list(B.neighbors(m))
                print(f"      - {m}: has {benchs}")
    
    # ==========================================
    # CHECK 3: Sparsity (primary benchmarks only)
    # ==========================================
    if verbose:
        print(f"\n--- CHECK 3: Sparsity (primary benchmarks only) ---")
    
    # Count edges only for primary benchmarks
    primary_edges = sum(1 for m in models_with_data for b in config.benchmarks if b in B.neighbors(m))
    matrix_size = len(models_with_data) * len(config.benchmarks)
    observed_count = primary_edges
    sparsity = 1.0 - (observed_count / matrix_size) if matrix_size > 0 else 0
    
    results['sparsity'] = sparsity
    results['observed_count'] = observed_count
    results['matrix_size'] = matrix_size
    
    if verbose:
        print(f"  Matrix size: {len(models_with_data)} models × {len(config.benchmarks)} benchmarks = {matrix_size}")
        print(f"  Observed scores: {observed_count}")
        print(f"  Sparsity: {sparsity:.1%}")
    
    if sparsity > 0.8:
        if verbose:
            print(f"  ⚠️  HIGH SPARSITY: Consider showing uncertainty intervals in visualizations")
        results['warnings'].append("High sparsity (>80%)")
    elif sparsity > 0.5:
        if verbose:
            print(f"  ⚠️  MODERATE SPARSITY: Bayesian shrinkage will help, but monitor uncertainty")
        results['warnings'].append("Moderate sparsity (>50%)")
    else:
        if verbose:
            print(f"  ✅ Low sparsity - good data coverage")
    
    # ==========================================
    # CHECK 4: Benchmark Coverage
    # ==========================================
    if verbose:
        print(f"\n--- CHECK 4: Benchmark Coverage ---")
    
    benchmark_coverage = {}
    
    # Primary benchmarks
    if verbose:
        print(f"  Primary benchmarks:")
    for bench in config.benchmarks:
        count = sum(1 for m in models if m.get(bench) is not None)
        pct = 100 * count / len(models) if models else 0
        benchmark_coverage[bench] = {'count': count, 'percent': pct, 'type': 'primary'}
        if verbose:
            print(f"    {bench}: {count}/{len(models)} ({pct:.0f}%)")
    
    # Auxiliary benchmarks
    if config.auxiliary:
        if verbose:
            print(f"  Auxiliary benchmarks (for covariance imputation):")
        for bench in config.auxiliary:
            count = sum(1 for m in models if m.get(bench) is not None)
            pct = 100 * count / len(models) if models else 0
            benchmark_coverage[bench] = {'count': count, 'percent': pct, 'type': 'auxiliary'}
            if verbose:
                print(f"    {bench}: {count}/{len(models)} ({pct:.0f}%)")
    
    results['benchmark_coverage'] = benchmark_coverage
    
    # ==========================================
    # CHECK 5: Score Output Coverage
    # ==========================================
    if verbose:
        print(f"\n--- CHECK 5: Output Score ({config.score_field}) ---")
    
    score_count = sum(1 for m in models if m.get(config.score_field) is not None)
    score_pct = 100 * score_count / len(models) if models else 0
    
    results['score_coverage'] = {'count': score_count, 'percent': score_pct}
    
    if verbose:
        print(f"  Models with {config.score_field}: {score_count}/{len(models)} ({score_pct:.0f}%)")
    
    return results


def print_summary(all_results: Dict[str, Dict]):
    """Print summary table of all composite scores."""
    print("\n" + "="*70)
    print("  SUMMARY: Bayesian Latent Factor Score Validation")
    print("="*70)
    
    print(f"\n{'Score':<8} {'Connected':<12} {'Anchors':<12} {'Sparsity':<12} {'Coverage':<12} {'Status':<10}")
    print("-"*70)
    
    for name, result in all_results.items():
        connected = "✅" if result.get('connectivity') == 'passed' else "❌"
        anchor_result = result.get('anchor_rule', '')
        if anchor_result == 'passed':
            anchors = "✅"
        elif anchor_result == 'passed_with_auxiliary':
            anchors = "✅ (aux)"  # Passed via auxiliary covariance
        else:
            anchors = "⚠️"
        sparsity = f"{result.get('sparsity', 0):.0%}"
        coverage = f"{result.get('score_coverage', {}).get('percent', 0):.0f}%"
        
        if result.get('passed') and not result.get('warnings'):
            status = "✅ OK"
        elif result.get('passed'):
            status = "⚠️ WARN"
        else:
            status = "❌ FAIL"
        
        print(f"{name:<8} {connected:<12} {anchors:<12} {sparsity:<12} {coverage:<12} {status:<10}")
    
    print()


def main():
    """Run validation on all composite scores."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate BLF composite scores")
    parser.add_argument('--score', choices=list(COMPOSITE_SCORES.keys()), 
                       help="Validate specific score only")
    parser.add_argument('--quiet', action='store_true',
                       help="Only show summary")
    args = parser.parse_args()
    
    print("Loading models cache...")
    models = load_models_cache()
    print(f"Loaded {len(models)} models\n")
    
    all_results = {}
    
    scores_to_check = [args.score] if args.score else COMPOSITE_SCORES.keys()
    
    for score_name in scores_to_check:
        config = COMPOSITE_SCORES[score_name]
        results = validate_composite_score(config, models, verbose=not args.quiet)
        all_results[score_name] = results
    
    print_summary(all_results)
    
    # Return exit code based on results
    if any(not r.get('passed') for r in all_results.values()):
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
