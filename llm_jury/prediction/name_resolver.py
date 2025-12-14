"""
Model name resolution using production model mappings.

Resolves model names between different naming conventions:
- OpenCompass names (e.g., 'gpt-4o-mini-2024-07-18')
- Cache names (e.g., 'GPT-4o mini')
"""

from typing import Optional, Dict

# Import the production mappings
from .models import OPENCOMPASS_TO_CACHE


class ModelNameResolver:
    """
    Resolve model names between different systems.
    
    Mappings are maintained in KDD/data/core_scripts/opencompass_name_mappings.py
    and cover 42+ models used in training.
    
    Example:
        >>> resolver = ModelNameResolver()
        >>> resolver.resolve('gpt-4o-mini-2024-07-18')
        'GPT-4o mini'
        >>> resolver.is_known('claude-3-7-sonnet-20250219')
        True
    """
    
    def __init__(self):
        """Initialize with production mappings."""
        self.mappings = OPENCOMPASS_TO_CACHE
        self._reverse_mappings = {v: k for k, v in self.mappings.items()}
    
    def resolve(self, model_name: str) -> str:
        """
        Resolve OpenCompass model name to cache name.
        
        Args:
            model_name: OpenCompass model name or cache name
        
        Returns:
            Cache name if found, else original name
        
        Example:
            >>> resolver = ModelNameResolver()
            >>> resolver.resolve('gpt-4o-mini-2024-07-18')
            'GPT-4o mini'
            >>> resolver.resolve('unknown-model')
            'unknown-model'
        """
        # Try direct mapping (OpenCompass → Cache)
        if model_name in self.mappings:
            return self.mappings[model_name]
        
        # Already in cache format (or unknown)
        return model_name
    
    def reverse_resolve(self, cache_name: str) -> Optional[str]:
        """
        Reverse resolve cache name to OpenCompass name.
        
        Args:
            cache_name: Cache model name
        
        Returns:
            OpenCompass name if found, else None
        
        Example:
            >>> resolver = ModelNameResolver()
            >>> resolver.reverse_resolve('GPT-4o mini')
            'gpt-4o-mini-2024-07-18'
        """
        return self._reverse_mappings.get(cache_name)
    
    def resolve_batch(self, model_names: list) -> Dict[str, str]:
        """
        Resolve multiple model names.
        
        Args:
            model_names: List of model names to resolve
        
        Returns:
            Dictionary mapping original_name -> resolved_name
        
        Example:
            >>> resolver = ModelNameResolver()
            >>> resolver.resolve_batch(['gpt-4o-mini-2024-07-18', 'GPT-4o'])
            {'gpt-4o-mini-2024-07-18': 'GPT-4o mini', 'GPT-4o': 'GPT-4o'}
        """
        return {name: self.resolve(name) for name in model_names}
    
    def is_known(self, model_name: str) -> bool:
        """
        Check if model name is in mappings.
        
        Args:
            model_name: Model name to check
        
        Returns:
            True if model is in OpenCompass or cache mappings
        
        Example:
            >>> resolver = ModelNameResolver()
            >>> resolver.is_known('gpt-4o-mini-2024-07-18')
            True
            >>> resolver.is_known('unknown-model-12345')
            False
        """
        return (model_name in self.mappings or 
                model_name in self._reverse_mappings)
    
    def get_all_mappings(self) -> Dict[str, str]:
        """
        Get all model name mappings.
        
        Returns:
            Dictionary of OpenCompass -> Cache mappings
        
        Example:
            >>> resolver = ModelNameResolver()
            >>> len(resolver.get_all_mappings())
            42
        """
        return self.mappings.copy()
    
    def get_all_cache_names(self) -> list:
        """Get list of all cache names."""
        return list(self._reverse_mappings.keys())
    
    def get_all_opencompass_names(self) -> list:
        """Get list of all OpenCompass names."""
        return list(self.mappings.keys())
    
    def print_summary(self):
        """Print summary of available mappings."""
        print("="*80)
        print("Model Name Resolver Summary")
        print("="*80)
        print(f"\nTotal mappings: {len(self.mappings)}")
        print(f"\nSample mappings (first 5):")
        print(f"{'OpenCompass Name':<40} {'Cache Name':<30}")
        print("-"*80)
        
        for i, (oc_name, cache_name) in enumerate(list(self.mappings.items())[:5]):
            print(f"{oc_name:<40} {cache_name:<30}")
        
        print(f"\n... and {len(self.mappings) - 5} more")
        print("\nUse resolver.get_all_mappings() to see all mappings")
        print("="*80)


# Singleton instance
_resolver = None

def get_resolver() -> ModelNameResolver:
    """
    Get or create the global name resolver.
    
    This is a singleton - the same resolver instance is returned
    across all calls to avoid reloading mappings.
    
    Returns:
        ModelNameResolver instance
    
    Example:
        >>> resolver = get_resolver()
        >>> resolver.resolve('gpt-4o-mini-2024-07-18')
        'GPT-4o mini'
    """
    global _resolver
    if _resolver is None:
        _resolver = ModelNameResolver()
    return _resolver


def resolve_name(model_name: str) -> str:
    """
    Convenience function to resolve a model name.
    
    Uses the global singleton resolver for efficiency.
    
    Args:
        model_name: Model name to resolve
    
    Returns:
        Resolved name (cache format)
    
    Example:
        >>> from llm_jury.prediction import resolve_name
        >>> resolve_name('gpt-4o-mini-2024-07-18')
        'GPT-4o mini'
    """
    return get_resolver().resolve(model_name)


if __name__ == '__main__':
    # When run directly, print summary
    resolver = ModelNameResolver()
    resolver.print_summary()
