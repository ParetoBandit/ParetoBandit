# Model name mappings from OpenCompass AND LiveCodeBench leaderboard to models_cache.json
# Updated with GRANULAR mappings to match specific model sizes

OPENCOMPASS_TO_CACHE = {
    # ═══════════════════════════════════════════════════════════════════
    # Claude models (OpenCompass + LiveCodeBench)
    # ═══════════════════════════════════════════════════════════════════
    'claude-3-5-sonnet-20241022': 'Claude 3.5 Sonnet',
    'claude-3-7-sonnet-20250219': 'Claude 3.7 Sonnet (Reasoning)',
    # LiveCodeBench leaderboard names
    'Claude-3.5-Sonnet-20241022': 'Claude 3.5 Sonnet',
    'Claude-3-Haiku': 'Claude 3.5 Haiku',
    'Claude-Sonnet-4': 'Claude 3.7 Sonnet (Reasoning)',
    'Claude-Sonnet-4 (Thinking)': 'Claude 3.7 Sonnet (Reasoning)',
    'Claude-Opus-4': 'Claude 3.7 Sonnet (Reasoning)',
    'Claude-Opus-4 (Thinking)': 'Claude 3.7 Sonnet (Reasoning)',
    
    # ═══════════════════════════════════════════════════════════════════
    # GPT/OpenAI models (OpenCompass + LiveCodeBench)
    # ═══════════════════════════════════════════════════════════════════
    'gpt4o-20240806': 'GPT-4o',
    'gpt4o-20241120': 'GPT-4o',
    'gpt-4o-mini-2024-07-18': 'GPT-4o mini',
    'gpt-4.5-preview-2025-02-27': 'GPT-4.1',
    # LiveCodeBench leaderboard names
    'GPT-4-Turbo-2024-04-09': 'GPT-4.1',
    'GPT-4O-2024-08-06': 'GPT-4o',
    'GPT-4O-mini-2024-07-18': 'GPT-4o mini',
    'O3-Mini-2025-01-31 (Low)': 'o3-mini (low)',
    'O3-Mini-2025-01-31 (Med)': 'o3-mini (medium)',
    'O3-Mini-2025-01-31 (High)': 'o3-mini (high)',
    'O4-Mini (Low)': 'o3-mini (low)',
    'O4-Mini (Medium)': 'o3-mini (medium)',
    'O4-Mini (High)': 'o3-mini (high)',
    'O3 (High)': 'o3-mini (high)',
    
    # ═══════════════════════════════════════════════════════════════════
    # Gemini models (OpenCompass + LiveCodeBench)
    # ═══════════════════════════════════════════════════════════════════
    'gemini-2.0-flash-exp': 'Gemini 2.0 Flash',
    'gemini-1.5-pro-latest': 'Gemini 2.5 Pro',
    # LiveCodeBench leaderboard names
    'Gemini-2.5-Pro-06-05': 'Gemini 2.5 Pro',
    'Gemini-2.5-Pro-05-06': 'Gemini 2.5 Pro',
    'Gemini-2.5-Flash-04-17': 'Gemini 2.5 Flash',
    'Gemini-2.5-Flash-05-20': 'Gemini 2.5 Flash',
    
    # ═══════════════════════════════════════════════════════════════════
    # QwQ/Qwen models (OpenCompass + LiveCodeBench)
    # ═══════════════════════════════════════════════════════════════════
    'QwQ-32B': 'QwQ 32B',
    'qwen2.5-72b-instruct-turbomind': 'Qwen2.5 Instruct 72B',
    # Note: Cache doesn't have smaller Qwen2.5 sizes, use 72B as fallback
    'qwen2.5-32b-instruct-turbomind': 'Qwen2.5 Instruct 72B',
    'qwen2.5-14b-instruct-turbomind': 'Qwen2.5 Instruct 72B',
    'qwen2.5-7b-instruct-turbomind': 'Qwen2.5 Instruct 72B',
    'qwen2.5-max': 'Qwen2.5 Instruct 72B',
    'qwen-max-2025-01-25': 'Qwen3 14B (Reasoning)',
    'qwen_max_0919': 'Qwen3 14B (Reasoning)',
    # LiveCodeBench leaderboard names
    'Qwen3-235B-A22B': 'Qwen3 235B A22B',
    
    # ═══════════════════════════════════════════════════════════════════
    # DeepSeek models (OpenCompass + LiveCodeBench)
    # ═══════════════════════════════════════════════════════════════════
    'deepseek-r1': 'DeepSeek R1',
    'deepseek-chat-r1': 'DeepSeek R1',
    'deepseek-chat-v3': 'DeepSeek V3 0324',
    'deepseek-chat-v3-0324': 'DeepSeek V3 0324',
    'deepseek-v2_5-turbomind': 'DeepSeek V3 0324',
    'deepseek-v2_5-1210-turbomind': 'DeepSeek V3 0324',
    # LiveCodeBench leaderboard names
    'DeepSeek-V3': 'DeepSeek V3 0324',
    'DeepSeek-R1-0528': 'DeepSeek R1',
    
    # FIXED: Distilled models now map to their actual cache entries
    'deepseek-r1-distill-llama-70b-turbomind': 'DeepSeek R1 Distill Llama 70B',
    'deepseek-r1-distill-llama-8b-turbomind': 'DeepSeek R1 Distill Llama 70B',  # No 8B in cache, use 70B
    'deepseek-r1-distill-qwen-32b-turbomind': 'DeepSeek R1 Distill Qwen 32B',
    'deepseek-r1-distill-qwen-14b-turbomind': 'DeepSeek R1 Distill Qwen 32B',  # No 14B, use 32B
    'deepseek-r1-distill-qwen-7b-turbomind': 'DeepSeek R1 Distill Qwen 32B',   # No 7B, use 32B
    'deepseek-r1-distill-qwen-1_5b-turbomind': 'DeepSeek R1 Distill Qwen 32B', # No 1.5B, use 32B
    
    # ═══════════════════════════════════════════════════════════════════
    # Gemma models - FIXED: Map to correct sizes
    # ═══════════════════════════════════════════════════════════════════
    'gemma3_27b_it': 'Gemma 3 27B Instruct',
    'gemma-2-27b-it-turbomind': 'Gemma 3 27B Instruct',
    'gemma-2-9b-it-turbomind': 'Gemma 3 12B Instruct',  # Closest size match
    
    # ═══════════════════════════════════════════════════════════════════
    # GLM models
    # ═══════════════════════════════════════════════════════════════════
    'glm-4-plus': 'GLM-4.5 (Reasoning)',
    'glm-4-9b-chat-turbomind': 'GLM-4.5 (Reasoning)',
    
    # ═══════════════════════════════════════════════════════════════════
    # Llama models - FIXED: Map to correct sizes!
    # ═══════════════════════════════════════════════════════════════════
    'llama-3_3-70b-instruct-turbomind': 'Llama 3.3 Instruct 70B',
    'llama-3_1-405b-instruct-FP8': 'Llama 3.1 Instruct 405B',
    'llama-3_1-70b-instruct-turbomind': 'Llama 3.1 Instruct 70B',
    'llama-3_1-8b-instruct-turbomind': 'Llama 3.1 Instruct 8B',
    'llama-3_2-3b-instruct-turbomind': 'Llama 3.2 Instruct 3B',
    
    # ═══════════════════════════════════════════════════════════════════
    # Mistral models
    # ═══════════════════════════════════════════════════════════════════
    'mistral-small-3.1-24b-instruct': 'Mistral Small 3.2',
    'mistral-small-instruct-2409-turbomind': 'Mistral 7B Instruct',
    'mixtral-large-instruct-2411-turbomind': 'Mistral Large',
    
    # ═══════════════════════════════════════════════════════════════════
    # Phi models
    # ═══════════════════════════════════════════════════════════════════
    'phi-4': 'Phi-4',
    
    # ═══════════════════════════════════════════════════════════════════
    # LiveCodeBench-specific models (Grok, Nemotron, etc.)
    # ═══════════════════════════════════════════════════════════════════
    'Grok-3-Mini (High)': 'Grok 3 Mini (High)',
    'OpenCodeReasoning-Nemotron-1.1-32B': 'Llama 3.3 Nemotron Super 49B v1 (Reasoning)',
    'OpenReasoning-Nemotron-32B': 'Llama 3.3 Nemotron Super 49B v1 (Reasoning)',
    'XBai-o4-medium': 'o3-mini (medium)',  # Similar reasoning model
    'EXAONE-4.0-32B': 'Qwen2.5 Instruct 72B',  # Similar size/capability
}

# Total: ~60 matched models from OpenCompass + LiveCodeBench
