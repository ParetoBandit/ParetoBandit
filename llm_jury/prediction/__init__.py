"""
Production intent prediction using KDD/data models.

This module wraps the validated production models from KDD/data/
for use in the llm_jury routing pipeline.

The models were trained on 113K instance-level examples from OpenCompass
benchmarks and use NVIDIA prompt complexity features combined with
model capability proxies.

Results:
- Reasoning: Test AUC 0.824, Transfer r=0.580***
- Coding: Test AUC 0.969, Transfer r=0.480***
- Summarization: Test AUC 0.896, Transfer r=0.744***
- RAG: Test AUC 0.779, Transfer r=0.453***

***p < 0.0001
"""

from .model_loader import load_model, load_all_models, get_model_info, get_all_model_info
from .name_resolver import ModelNameResolver, get_resolver, resolve_name

__all__ = [
    # Model loading
    'load_model',
    'load_all_models',
    'get_model_info',
    'get_all_model_info',
    # Name resolution
    'ModelNameResolver',
    'get_resolver',
    'resolve_name',
]

__version__ = '1.0.0'
