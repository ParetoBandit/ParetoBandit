"""
Updated version - to replace official_sources_scraper.py
This version uses OpenRouter canonical names
"""

# This helper function will be added to the existing file
def process_curated_benchmarks_with_matching(benchmarks_raw, known_models):
    """
    Process curated benchmarks and match to OpenRouter canonical names.
    
    Args:
        benchmarks_raw: List of dicts with 'name_variants' or 'model_name'
        known_models: List of canonical model names from OpenRouter
        
    Returns:
        List of dicts with matched 'model_name'
    """
    def normalize_name(name):
        return name.lower().replace('-', '').replace('_', '').replace('/', '').replace('.', '')
    
    def match_model_name(name_variants, known_models):
        if not known_models:
            return name_variants[0] if isinstance(name_variants, list) else name_variants
        
        # Handle both list and string
        if not isinstance(name_variants, list):
            name_variants = [name_variants]
        
        # Normalize known models
        normalized_known = {normalize_name(name): name for name in known_models}
        
        # Try each variant
        for variant in name_variants:
            normalized_variant = normalize_name(variant)
            
            # Exact match
            if normalized_variant in normalized_known:
                return normalized_known[normalized_variant]
            
            # Partial match
            for norm_known, original_known in normalized_known.items():
                if normalized_variant in norm_known or norm_known in normalized_variant:
                    return original_known
        
        return name_variants[0]
    
    results = []
    for benchmark in benchmarks_raw:
        # Get name variants
        name_variants = benchmark.pop('name_variants', None) or benchmark.pop('model_name', None)
        
        if not name_variants:
            continue
        
        # Match to canonical name
        matched_name = match_model_name(name_variants, known_models)
        
        # Create result with matched name
        result = {'model_name': matched_name}
        result.update(benchmark)
        results.append(result)
    
    return results

