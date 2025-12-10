"""
Aggregate scraper that combines data from all sources into comprehensive benchmark dataset.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import logging

# ChatbotArenaScraper removed - use manual curation via update_arena_rankings.py
from .artificial_analysis_scraper import ArtificialAnalysisScraper
from .openrouter_scraper import OpenRouterScraper
from .official_sources_scraper import OfficialSourcesScraper
from .huggingface_scraper import HuggingFaceLeaderboardScraper
from .benchmark_estimator import BenchmarkEstimator

logger = logging.getLogger(__name__)


class ComprehensiveBenchmarkAggregator:
    """
    Aggregates data from multiple sources into a comprehensive benchmark dataset.
    
    Combines:
    - Official benchmarks (MMLU, GPQA, MATH, IFEval)
    - Arena ratings (Elo, MT-Bench)
    - Performance metrics (latency, throughput)
    - Pricing and specifications
    """
    
    def __init__(self, cache_dir: str = ".cache/scraped_benchmarks"):
        """
        Initialize aggregator.
        
        Args:
            cache_dir: Directory to store scraped data cache
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize scrapers (all will be updated with known models after OpenRouter scrape)
        self.scrapers = {
            'openrouter': OpenRouterScraper(),  # Scrape this first to get canonical names
            'huggingface': HuggingFaceLeaderboardScraper(),  # Add HF benchmarks with fuzzy matching
            # 'arena': ChatbotArenaScraper(),  # Removed - use manual curation
            'performance': ArtificialAnalysisScraper(),
            'official': OfficialSourcesScraper(),
        }
    
    def collect_all_data(self, use_cache: bool = True) -> pd.DataFrame:
        """
        Collect data from all sources and merge into comprehensive dataset.
        
        Args:
            use_cache: If True, use cached data if available and recent
            
        Returns:
            DataFrame with comprehensive model benchmarks
        """
        logger.info("="*80)
        logger.info("🚀 COMPREHENSIVE BENCHMARK DATA COLLECTION")
        logger.info("="*80)
        
        # Check cache
        cache_file = self.cache_dir / "comprehensive_benchmarks.json"
        if use_cache and cache_file.exists():
            cache_age_hours = (datetime.now().timestamp() - cache_file.stat().st_mtime) / 3600
            if cache_age_hours < 24:  # Cache valid for 24 hours
                logger.info(f"📦 Using cached data (age: {cache_age_hours:.1f} hours)")
                return self._load_from_cache(cache_file)
        
        # Collect from all sources
        all_data = {}
        
        logger.info("\n📊 Collecting from multiple sources...")
        
        # Step 1: Scrape OpenRouter first to get canonical model names
        logger.info(f"\n  ▶ {self.scrapers['openrouter'].get_source_name()} (Primary)...")
        try:
            openrouter_data = self.scrapers['openrouter'].scrape()
            all_data['openrouter'] = openrouter_data
            logger.info(f"    ✅ Collected {len(openrouter_data)} entries")
            
            # Extract model names for other scrapers
            known_models = [m.get('model_name') for m in openrouter_data if m.get('model_name')]
            logger.info(f"    📋 Extracted {len(known_models)} canonical model names")
            
            # Update all scrapers with known models
            for scraper_name, scraper in self.scrapers.items():
                if scraper_name != 'openrouter' and hasattr(scraper, 'known_models'):
                    scraper.known_models = known_models
            
            logger.info(f"    🔗 Updated all scrapers with OpenRouter model names")
            
        except Exception as e:
            logger.error(f"    ❌ Failed: {e}")
            all_data['openrouter'] = []
        
        # Step 2: Scrape other sources
        for name, scraper in self.scrapers.items():
            if name == 'openrouter':
                continue  # Already scraped
            
            logger.info(f"\n  ▶ {scraper.get_source_name()}...")
            try:
                data = scraper.scrape()
                all_data[name] = data
                logger.info(f"    ✅ Collected {len(data)} entries")
            except Exception as e:
                logger.error(f"    ❌ Failed: {e}")
                all_data[name] = []
        
        # Merge all data sources
        logger.info("\n🔄 Merging data from all sources...")
        merged_df = self._merge_all_sources(all_data)
        
        # Enrich with estimated benchmarks
        logger.info("\n🤖 Enriching with estimated benchmarks...")
        merged_df = self._enrich_with_estimates(merged_df, all_data.get('huggingface', []))
        
        # Save to cache
        self._save_to_cache(merged_df, cache_file)
        
        logger.info(f"\n✅ Final dataset: {len(merged_df)} models with comprehensive benchmarks")
        logger.info("="*80)
        
        return merged_df
    
    def _merge_all_sources(self, all_data: Dict[str, List[Dict]]) -> pd.DataFrame:
        """
        Merge data from all sources into comprehensive dataset.
        
        Strategy:
        1. Start with OpenRouter data (most comprehensive model list)
        2. Merge official benchmarks (highest priority for proprietary models)
        3. Merge HuggingFace benchmarks (for open source models)
        4. Add performance metrics
        5. Add arena ratings
        6. Deduplicate and normalize model names
        """
        # Convert each source to DataFrame
        dfs = {}
        for source_name, data in all_data.items():
            if data:
                dfs[source_name] = pd.DataFrame(data)
        
        # Start with OpenRouter data as base (most comprehensive model list)
        if 'openrouter' in dfs and not dfs['openrouter'].empty:
            merged = dfs['openrouter'].copy()
            logger.info(f"  Starting with {len(merged)} OpenRouter models")
        else:
            merged = pd.DataFrame()
            logger.warning("  No OpenRouter data available")
        
        # Merge official benchmarks (priority for proprietary models)
        if 'official' in dfs and not dfs['official'].empty:
            official_df = dfs['official']
            merged = self._merge_dataframes(
                merged, official_df,
                on='model_name',
                how='left',  # Keep all OpenRouter models
                suffixes=('', '_official')
            )
            logger.info(f"  After official benchmarks merge: {len(merged)} models, {(merged['mmlu_score'] > 0).sum()} with MMLU")
        
        # Merge HuggingFace benchmarks (for open source models)
        if 'huggingface' in dfs and not dfs['huggingface'].empty:
            hf_df = dfs['huggingface']
            # Only use HF data for models that exist in OpenRouter
            # This prevents adding thousands of unknown models
            merged = self._merge_dataframes(
                merged, hf_df,
                on='model_name',
                how='left',  # ONLY keep OpenRouter models, add HF data where names match
                suffixes=('', '_hf')
            )
            # Count models that got benchmark data from HF
            mmlu_count = (merged['mmlu_score'].fillna(0) > 0).sum()
            logger.info(f"  After HuggingFace merge: {len(merged)} models, {mmlu_count} with MMLU")
        
        # Merge performance metrics
        if 'performance' in dfs and not dfs['performance'].empty:
            perf_df = dfs['performance']
            merged = self._merge_dataframes(
                merged, perf_df,
                on='model_name',
                how='left',  # Keep all OpenRouter models
                suffixes=('', '_perf')
            )
            logger.info(f"  After performance merge: {len(merged)} models")
        
        # Merge arena ratings
        if 'arena' in dfs and not dfs['arena'].empty:
            arena_df = dfs['arena']
            merged = self._merge_dataframes(
                merged, arena_df,
                on='model_name',
                how='left',  # Keep all OpenRouter models
                suffixes=('', '_arena')
            )
            logger.info(f"  After arena merge: {len(merged)} models")
        
        # Normalize and clean
        merged = self._normalize_dataset(merged)
        
        return merged
    
    def _enrich_with_estimates(self, df: pd.DataFrame, hf_data: List[Dict]) -> pd.DataFrame:
        """
        Enrich models with estimated benchmarks based on similar models.
        
        Args:
            df: DataFrame with models (some with benchmarks, some without)
            hf_data: HuggingFace benchmark data to use as reference
            
        Returns:
            DataFrame with estimated benchmarks added
        """
        if df.empty or not hf_data:
            logger.warning("  No data to enrich")
            return df
        
        # Count models without benchmarks
        models_without = (df['mmlu_score'].fillna(0) == 0).sum()
        logger.info(f"  Found {models_without} models without benchmark data")
        
        if models_without == 0:
            logger.info("  All models already have benchmarks!")
            return df
        
        # Create estimator with HF data as reference
        estimator = BenchmarkEstimator(hf_data)
        
        # Convert df to list of dicts for enrichment
        models_list = df.to_dict('records')
        
        # Enrich with estimates
        enriched_models = estimator.enrich_models(models_list)
        
        # Convert back to DataFrame
        enriched_df = pd.DataFrame(enriched_models)
        
        # Count how many were enriched
        models_with_estimates = enriched_df.get('estimated', pd.Series([False]*len(enriched_df))).sum()
        logger.info(f"  ✅ Added estimated benchmarks for {models_with_estimates} models")
        
        return enriched_df
    
    def _merge_dataframes(self, left: pd.DataFrame, right: pd.DataFrame, 
                         on: str, how: str = 'outer', 
                         suffixes: tuple = ('', '_y')) -> pd.DataFrame:
        """Merge two dataframes with fuzzy model name matching."""
        if left.empty:
            return right
        if right.empty:
            return left
        
        # Normalize model names for matching
        left_normalized = left.copy()
        right_normalized = right.copy()
        
        left_normalized['_normalized_name'] = left_normalized[on].apply(self._normalize_model_name)
        right_normalized['_normalized_name'] = right_normalized[on].apply(self._normalize_model_name)
        
        # Merge on normalized names
        merged = pd.merge(
            left_normalized,
            right_normalized,
            on='_normalized_name',
            how=how,
            suffixes=suffixes
        )
        
        # Prefer original name from left, fallback to right
        if f'{on}{suffixes[0]}' in merged.columns and f'{on}{suffixes[1]}' in merged.columns:
            merged[on] = merged[f'{on}{suffixes[0]}'].fillna(merged[f'{on}{suffixes[1]}'])
        
        # Drop temporary columns
        merged = merged.drop(columns=['_normalized_name'], errors='ignore')
        
        return merged
    
    def _normalize_model_name(self, name: str) -> str:
        """
        Normalize model name for fuzzy matching.
        
        Examples:
        - "gpt-4o" -> "gpt4o"
        - "openai/gpt-4o" -> "gpt4o"
        - "GPT-4-Turbo" -> "gpt4turbo"
        """
        if pd.isna(name):
            return ""
        
        name = str(name).lower()
        
        # Remove provider prefix
        if '/' in name:
            name = name.split('/')[-1]
        
        # Remove special characters
        name = name.replace('-', '').replace('_', '').replace('.', '')
        
        # Remove common suffixes
        for suffix in ['instruct', 'chat', 'preview', 'exp', 'experimental']:
            name = name.replace(suffix, '')
        
        return name.strip()
    
    def _normalize_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize and clean the merged dataset."""
        if df.empty:
            return df
        
        # CRITICAL: Only keep models that exist in OpenRouter
        # Filter out HuggingFace-only models that don't have pricing
        if 'input_cost_per_m' in df.columns:
            # Keep models that have pricing data (from OpenRouter)
            has_pricing = df['input_cost_per_m'].fillna(0) > 0
            df = df[has_pricing]
            logger.info(f"  Filtered to {len(df)} OpenRouter models (removed HF-only models)")
        
        # Remove duplicates based on model_name
        if 'model_name' in df.columns:
            df = df.drop_duplicates(subset=['model_name'], keep='first')
            logger.info(f"  After deduplication: {len(df)} unique models")
        
        # Consolidate duplicate columns (e.g., source, source_or, source_perf, mmlu_score_hf)
        # First, handle benchmark columns from HuggingFace
        benchmark_cols = ['mmlu_score', 'gpqa_score', 'math_score', 'ifeval_score', 'param_count_b']
        for base_col in benchmark_cols:
            hf_col = f"{base_col}_hf"
            if hf_col in df.columns and base_col in df.columns:
                # Fill missing values in base column with HF data
                df[base_col] = df[base_col].fillna(df[hf_col])
                # Also fill zeros with HF data if HF has better data
                mask = (df[base_col] == 0) & (df[hf_col] > 0)
                df.loc[mask, base_col] = df.loc[mask, hf_col]
                logger.info(f"    Coalesced {base_col}: filled {mask.sum()} values from HuggingFace")
        
        # Then handle other duplicate columns
        for base_col in ['source', 'model_id', 'provider']:
            cols_to_merge = [c for c in df.columns if c.startswith(base_col)]
            if len(cols_to_merge) > 1:
                # Coalesce: take first non-null value
                df[base_col] = df[cols_to_merge].bfill(axis=1).iloc[:, 0]
                # Drop duplicates
                df = df.drop(columns=[c for c in cols_to_merge if c != base_col], errors='ignore')
        
        # Ensure all benchmark columns exist with defaults
        benchmark_columns = {
            'mmlu_score': 0.0,
            'gpqa_score': 0.0,
            'math_score': 0.0,
            'ifeval_score': 0.0,
            'humaneval_score': 0.0,
            'tool_use_ability': 0.0,
            'arena_elo': 0,
            'mt_bench_score': 0.0,
            'latency_ms': 0.0,
            'throughput_tps': 0.0,
            'quality_index': 0.0,
            'context_length': 0,
            'param_count_b': 0.0,
            'input_cost_per_m': 0.0,
            'output_cost_per_m': 0.0,
        }
        
        for col, default in benchmark_columns.items():
            if col not in df.columns:
                df[col] = default
            else:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(default)
        
        # Calculate composite scores
        df['quality_score'] = (
            df['mmlu_score'] * 0.3 +
            df['gpqa_score'] * 0.3 +
            df['math_score'] * 0.2 +
            df['ifeval_score'] * 0.2
        )
        
        df['blended_cost'] = df['input_cost_per_m'] * 0.75 + df['output_cost_per_m'] * 0.25
        
        # Sort by quality score
        df = df.sort_values('quality_score', ascending=False)
        
        return df
    
    def _save_to_cache(self, df: pd.DataFrame, cache_file: Path):
        """Save dataset to cache."""
        try:
            # Convert to records for JSON serialization
            records = df.to_dict('records')
            
            cache_data = {
                'metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'total_models': len(df),
                    'sources': list(self.scrapers.keys()),
                },
                'models': records
            }
            
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            
            logger.info(f"💾 Saved to cache: {cache_file}")
            
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
    
    def _load_from_cache(self, cache_file: Path) -> pd.DataFrame:
        """Load dataset from cache."""
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
            
            df = pd.DataFrame(data['models'])
            logger.info(f"✅ Loaded {len(df)} models from cache")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load cache: {e}")
            return pd.DataFrame()
    
    def export_to_csv(self, df: pd.DataFrame, output_file: str = "comprehensive_benchmarks.csv"):
        """Export dataset to CSV."""
        try:
            df.to_csv(output_file, index=False)
            logger.info(f"📊 Exported to: {output_file}")
        except Exception as e:
            logger.error(f"Failed to export CSV: {e}")
    
    def print_summary(self, df: pd.DataFrame):
        """Print summary statistics of the collected data."""
        if df.empty:
            logger.warning("No data to summarize")
            return
        
        print("\n" + "="*80)
        print("📈 COMPREHENSIVE BENCHMARK DATASET SUMMARY")
        print("="*80)
        
        print(f"\n📊 Coverage:")
        print(f"  • Total models: {len(df)}")
        print(f"  • Models with MMLU: {(df['mmlu_score'] > 0).sum()}")
        print(f"  • Models with Arena Elo: {(df['arena_elo'] > 0).sum()}")
        print(f"  • Models with latency data: {(df['latency_ms'] > 0).sum()}")
        print(f"  • Models with pricing: {(df['input_cost_per_m'] > 0).sum()}")
        
        print(f"\n🏆 Top 5 Models by Quality Score:")
        top_models = df.nlargest(5, 'quality_score')[['model_name', 'quality_score', 'mmlu_score', 'arena_elo']]
        for idx, row in top_models.iterrows():
            print(f"  {row['model_name']:<30} Quality: {row['quality_score']:.1f}  MMLU: {row['mmlu_score']:.1f}  Elo: {int(row['arena_elo'])}")
        
        print(f"\n💰 Price Range:")
        print(f"  • Min: ${df[df['blended_cost'] > 0]['blended_cost'].min():.4f} per 1M tokens")
        print(f"  • Median: ${df[df['blended_cost'] > 0]['blended_cost'].median():.4f} per 1M tokens")
        print(f"  • Max: ${df['blended_cost'].max():.2f} per 1M tokens")
        
        print(f"\n⚡ Performance Range:")
        print(f"  • Fastest latency: {df[df['latency_ms'] > 0]['latency_ms'].min():.0f} ms")
        print(f"  • Median latency: {df[df['latency_ms'] > 0]['latency_ms'].median():.0f} ms")
        print(f"  • Slowest latency: {df[df['latency_ms'] > 0]['latency_ms'].max():.0f} ms")
        
        print("="*80 + "\n")

