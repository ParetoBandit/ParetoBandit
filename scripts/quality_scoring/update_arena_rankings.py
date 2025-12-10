#!/usr/bin/env python3
"""
Update models cache with LMArena leaderboard rankings.

Data source: https://lmarena.ai/leaderboard
Extracted: December 2024

This script adds category-specific Arena rankings to the models cache:
- arena_rank_overall: Overall Arena ranking
- arena_rank_expert: Expert category ranking
- arena_rank_hard: Hard Prompts ranking
- arena_rank_coding: Coding ranking
- arena_rank_math: Math ranking
- arena_rank_creative: Creative Writing ranking
- arena_rank_instruction: Instruction Following ranking
- arena_rank_longer: Longer Query ranking
- arena_elo_text: Text Arena ELO score
"""

import json
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

# LMArena Text ELO scores (from leaderboard, December 2024)
ARENA_ELO_SCORES = {
    "gemini-3-pro": 1491,
    "grok-4.1-thinking": 1481,
    "claude-opus-4-5-20251101-thinking-32k": 1471,
    "grok-4.1": 1463,
    "claude-opus-4-5-20251101": 1462,
    "gpt-5.1-high": 1457,
    "gemini-2.5-pro": 1451,
    "claude-sonnet-4-5-20250929-thinking-32k": 1448,
    "claude-opus-4-1-20250805-thinking-16k": 1448,
    "claude-sonnet-4-5-20250929": 1445,
}

# Arena Overview rankings by category
# Format: model_id -> (overall, expert, hard, coding, math, creative, instruction, longer)
# "-" in the original data means no ranking available (stored as None)
ARENA_RANKINGS = {
    "gemini-3-pro": (1, 4, 1, 2, 2, 1, 1, 2),
    "grok-4.1-thinking": (2, 5, 4, 6, 8, 5, 10, 11),
    "claude-opus-4-5-20251101-thinking-32k": (3, 2, 2, 1, 4, 4, 2, 1),
    "grok-4.1": (4, 25, 7, 11, 16, 14, 14, 12),
    "claude-opus-4-5-20251101": (5, 1, 3, 5, 1, 3, 3, 3),
    "gpt-5.1-high": (6, 7, 8, 10, 3, 8, 8, 9),
    "gemini-2.5-pro": (7, 10, 11, 22, 7, 2, 9, 8),
    "claude-sonnet-4-5-20250929-thinking-32k": (8, 3, 5, 3, 5, 7, 5, 4),
    "claude-opus-4-1-20250805-thinking-16k": (9, 8, 6, 4, 9, 6, 4, 5),
    "claude-sonnet-4-5-20250929": (10, 6, 9, 7, 17, 9, 6, 6),
    "gpt-4.5-preview-2025-02-27": (11, 36, 28, 35, 38, 11, 12, 17),
    "claude-opus-4-1-20250805": (12, 14, 10, 8, 14, 10, 7, 7),
    "chatgpt-4o-latest-20250326": (13, 39, 14, 27, 55, 13, 16, 19),
    "gpt-5-high": (14, 13, 16, 21, 11, 41, 22, 42),
    "gpt-5.1": (15, 11, 15, 16, 52, 20, 15, 16),
    "o3-2025-04-16": (16, 18, 26, 34, 6, 40, 43, 49),
    "qwen3-max-preview": (17, 9, 12, 13, 10, 27, 13, 13),
    "kimi-k2-thinking-turbo": (18, 19, 18, 14, 20, 22, 19, 28),
    "grok-4-1-fast-reasoning": (19, 16, 29, 40, 41, 17, 34, 46),
    "glm-4.6": (20, 23, 23, 28, 18, 23, 17, 25),
    "gpt-5-chat": (21, 22, 19, 31, 36, 37, 23, 23),
    "qwen3-max-2025-09-23": (22, 37, 21, 17, 13, 28, 24, 26),
    "claude-opus-4-20250514-thinking-16k": (23, 20, 13, 9, 27, 12, 11, 10),
    "deepseek-v3.2-exp": (24, 41, 20, 24, 35, 21, 25, 18),
    "mistral-large-3": (25, 59, 22, 12, 47, 58, 32, 36),
    "qwen3-235b-a22b-instruct-2507": (26, 17, 17, 20, 19, 45, 20, 22),
    "deepseek-v3.2-exp-thinking": (27, 29, 24, 19, 22, 30, 21, 27),
    "grok-4-fast-chat": (28, 38, 40, 39, 25, 39, 42, 33),
    "deepseek-v3.2-thinking": (29, 70, 43, 36, 28, 38, 37, 43),
    "kimi-k2-0905-preview": (30, 40, 32, 25, 34, 42, 56, 57),
    "deepseek-r1-0528": (31, 44, 35, 29, 63, 35, 50, 52),
    "ernie-5.0-preview-1022": (32, 21, 45, 57, 33, 15, 45, 40),
    "kimi-k2-0711-preview": (33, 46, 39, 32, 67, 55, 66, 61),
    "deepseek-v3.1": (34, 34, 38, 47, 31, 32, 39, 32),
    "deepseek-v3.1-thinking": (35, 33, 30, 37, 26, 19, 18, 14),
    "deepseek-v3.1-terminus": (36, None, 47, 56, 61, 18, 48, 45),
    "qwen3-vl-235b-a22b-instruct": (37, 24, 27, 26, 39, 64, 27, 35),
    "deepseek-v3.1-terminus-thinking": (38, None, 25, 33, 32, 51, 26, 20),
    "deepseek-v3.2": (39, 45, 34, 46, 15, 31, 30, 30),
    "claude-opus-4-20250514": (40, 35, 33, 30, 57, 16, 31, 15),
    "gpt-4.1-2025-04-14": (41, 55, 42, 42, 79, 24, 46, 34),
    "mistral-medium-2508": (42, 52, 37, 44, 50, 47, 44, 47),
    "grok-3-preview-02-24": (43, 51, 46, 52, 76, 26, 35, 29),
    "grok-4-0709": (44, 31, 49, 58, 12, 29, 47, 44),
    "glm-4.5": (45, 26, 36, 38, 30, 52, 33, 38),
    "gemini-2.5-flash": (46, 32, 59, 74, 42, 25, 40, 39),
    "gemini-2.5-flash-preview-09-2025": (47, 15, 50, 73, 24, 43, 41, 41),
    "grok-4-fast-reasoning": (48, 43, 61, 54, 44, 48, 55, 50),
    "claude-haiku-4-5-20251001": (49, 27, 31, 15, 68, 49, 29, 24),
    "o1-2024-12-17": (50, 56, 55, 64, 45, 46, 38, 48),
    "qwen3-next-80b-a3b-instruct": (51, 57, 48, 49, 23, 97, 57, 60),
    "longcat-flash-chat": (52, 42, 44, 18, 21, 84, 51, 67),
    "claude-sonnet-4-20250514-thinking-32k": (53, 30, 41, 23, 48, 33, 28, 21),
    "qwen3-235b-a22b-no-thinking": (54, 62, 52, 50, 62, 60, 60, 53),
    "qwen3-235b-a22b-thinking-2507": (55, 12, 51, 53, 53, 56, 52, 56),
    "deepseek-r1": (56, 63, 54, 51, 43, 54, 49, 58),
    "qwen3-vl-235b-a22b-thinking": (57, 28, 53, 41, 40, 76, 58, 55),
    "gpt-5-mini-high": (58, 49, 63, 65, 37, 87, 64, 78),
    "deepseek-v3-0324": (59, 65, 64, 71, 78, 34, 65, 65),
    "hunyuan-vision-1.5-thinking": (60, None, 56, 59, None, 63, 54, 59),
    "o4-mini-2025-04-16": (61, 47, 65, 66, 29, 78, 71, 86),
    "mai-1-preview": (62, 53, 66, 68, 60, 68, 67, 63),
    "claude-sonnet-4-20250514": (63, 60, 57, 48, 69, 44, 53, 37),
    "o1-preview": (64, 76, 73, 76, 70, 61, 63, 75),
    "claude-3-7-sonnet-20250219-thinking-32k": (65, 48, 58, 45, 74, 36, 36, 31),
    "qwen3-coder-480b-a35b-instruct": (66, 75, 60, 43, 77, 66, 59, 54),
    "hunyuan-t1-20250711": (67, 58, 69, 86, 51, 50, 62, 69),
    "mistral-medium-2505": (68, 74, 70, 67, 95, 67, 76, 66),
    "qwen3-30b-a3b-instruct-2507": (69, 69, 62, 55, 72, 90, 72, 72),
    "gpt-4.1-mini-2025-04-14": (70, 73, 67, 60, 93, 74, 69, 68),
    "hunyuan-turbos-20250416": (71, 92, 77, 90, 97, 62, 83, 76),
    "gemini-2.5-flash-lite-preview-09-2025-no-thinking": (72, 67, 74, 89, 80, 65, 73, 64),
    "gemini-2.5-flash-lite-preview-06-17-thinking": (73, 81, 79, 103, 82, 57, 68, 71),
    "qwen3-235b-a22b": (74, 72, 75, 63, 58, 88, 79, 74),
    "qwen2.5-max": (75, 84, 80, 84, 84, 70, 80, 70),
    "claude-3-5-sonnet-20241022": (76, 83, 71, 61, 98, 59, 70, 62),
    "claude-3-7-sonnet-20250219": (77, 71, 72, 69, 89, 53, 61, 51),
    "glm-4.5-air": (78, 66, 76, 70, 64, 89, 75, 73),
    "qwen3-next-80b-a3b-thinking": (79, 68, 78, 72, 59, 94, 78, 82),
    "minimax-m1": (80, 78, 81, 75, 56, 93, 85, 83),
    "gemma-3-27b-it": (81, 100, 93, 126, 105, 72, 89, 87),
    "o3-mini-high": (82, 64, 68, 62, 46, 101, 74, 81),
    "grok-3-mini-high": (83, 54, 83, 95, 65, 86, 77, 77),
    "gemini-2.0-flash-001": (84, 93, 97, 122, 92, 73, 86, 89),
    "deepseek-v3": (85, 95, 108, 97, 112, 71, 91, 80),
    "grok-3-mini-beta": (86, 77, 88, 100, 81, 79, 81, 84),
    "mistral-small-2506": (87, 102, 85, 79, 99, 92, 95, 91),
    "gemini-2.0-flash-lite-preview-02-05": (88, 103, 109, 145, 103, 75, 100, 101),
    "gpt-oss-120b": (89, 86, 96, 93, 71, 137, 103, 131),
    "command-a-03-2025": (90, 98, 92, 94, 114, 82, 92, 85),
    "glm-4.5v": (91, 50, 87, 83, 88, 102, 93, 110),
    "gemini-1.5-pro-002": (92, 101, 107, 132, 101, 69, 94, 94),
    "amazon-nova-experimental-chat-10-20": (93, 79, 84, 80, 54, 139, 82, 88),
    "o3-mini": (94, 85, 89, 77, 73, 110, 90, 92),
    "hunyuan-turbos-20250226": (95, None, 95, 85, 123, 112, 87, 95),
    "ling-flash-2.0": (96, 89, 91, 78, 91, 148, 109, 125),
    "minimax-m2": (97, 99, 90, 107, 87, 129, 96, 100),
    "step-3": (98, 90, 82, 82, 83, 107, 88, 97),
    "llama-3.1-nemotron-ultra-253b-v1": (99, None, 86, 92, 75, 83, 84, 104),
    "amazon-nova-experimental-chat-10-09": (100, None, 102, 96, None, 120, 110, 112),
    "gpt-4o-2024-05-13": (101, 119, 116, 116, 119, 80, 106, 124),
    "qwen3-32b": (102, 61, 94, 81, 49, 108, 101, 93),
    "qwen-plus-0125": (103, 82, 106, 106, 104, 100, 98, 90),
    "glm-4-plus-0111": (104, 118, 130, 153, 125, 98, 113, 111),
    "claude-3-5-sonnet-20240620": (105, 97, 98, 87, 100, 111, 97, 96),
    "gemma-3-12b-it": (106, 147, 121, 158, 107, 85, 108, 99),
    "nvidia-llama-3.3-nemotron-super-49b-v1.5": (107, 80, 99, 88, 66, 106, 107, 106),
    "hunyuan-turbo-0110": (108, None, 110, 112, 145, 116, 119, 105),
    "gpt-5-nano-high": (109, 91, 104, 105, 94, 162, 102, 121),
    "llama-3.1-405b-instruct-bf16": (110, 121, 113, 109, 109, 114, 121, 130),
    "o1-mini": (111, 94, 100, 98, 86, 140, 99, 103),
    "gpt-4o-2024-08-06": (112, 120, 131, 128, 116, 91, 112, 115),
    "grok-2-2024-08-13": (113, 114, 134, 130, 132, 99, 125, 120),
    "qwq-32b": (114, 88, 103, 101, 85, 122, 104, 113),
    "gemini-advanced-0514": (115, 137, 133, 149, 122, 77, 117, 136),
    "llama-3.1-405b-instruct-fp8": (116, 109, 118, 120, 108, 109, 120, 140),
    "step-2-16k-exp-202412": (117, 106, 117, 114, 110, 81, 114, 108),
    "yi-lightning": (118, 104, 114, 115, 117, 119, 127, 133),
    "llama-4-maverick-17b-128e-instruct": (119, 105, 115, 111, 106, 104, 118, 119),
    "qwen3-30b-a3b": (120, 96, 111, 102, 90, 133, 122, 109),
    "llama-3.3-nemotron-49b-super-v1": (121, None, 101, 123, None, 121, 105, 117),
    "hunyuan-large-2025-02-10": (122, 108, 126, 121, 130, 113, 115, 79),
    "gpt-4-turbo-2024-04-09": (123, 131, 144, 142, 127, 95, 133, 144),
    "claude-3-5-haiku-20241022": (124, 115, 112, 104, 141, 115, 124, 107),
    "llama-4-scout-17b-16e-instruct": (125, 122, 125, 124, 111, 125, 136, 126),
    "deepseek-v2.5-1210": (126, 123, 128, 108, 131, 103, 116, 118),
    "claude-3-opus-20240229": (127, 111, 127, 135, 115, 130, 126, 129),
    "gemini-1.5-pro-001": (128, 112, 129, 141, 129, 96, 128, 98),
    "gpt-4.1-nano-2025-04-14": (129, 113, 124, 110, 148, 105, 134, 128),
    "ring-flash-2.0": (130, 87, 105, 91, 96, 160, 111, 122),
    "step-1o-turbo-202506": (131, 128, 122, 133, 120, 126, 123, 102),
    "llama-3.3-70b-instruct": (132, 132, 137, 144, 128, 132, 144, 148),
    "gemma-3n-e4b-it": (133, 143, 145, 162, 157, 117, 149, 143),
    "glm-4-plus": (134, 125, 141, 138, 139, 127, 135, 134),
    "gpt-oss-20b": (135, 110, 136, 113, 102, 171, 152, 150),
    "qwen-max-0919": (136, 126, 142, 137, 133, 136, 132, 132),
    "gpt-4o-mini-2024-07-18": (137, 141, 149, 140, 147, 123, 142, 135),
    "qwen2.5-plus-1127": (138, 107, 132, 131, 118, 146, 139, 149),
    "gpt-4-1106-preview": (139, 140, 146, 150, 121, 135, 140, 156),
    "mistral-large-2407": (140, 127, 139, 136, 137, 131, 137, 153),
    "gpt-4-0125-preview": (141, 145, 152, 155, 124, 141, 148, 151),
    "athene-v2-chat": (142, 116, 120, 117, 113, 166, 131, 138),
    "olmo-3-32b-think": (143, None, 123, 127, None, 153, 129, 114),
    "mercury": (144, None, 135, 118, None, 172, 155, 147),
    "hunyuan-standard-2025-02-10": (145, 138, 150, 154, 134, 145, 151, 123),
    "gemini-1.5-flash-002": (146, 144, 156, 160, 135, 118, 146, 137),
    "grok-2-mini-2024-08-13": (147, 136, 155, 151, 150, 149, 150, 142),
    "deepseek-v2.5": (148, 130, 138, 119, 136, 152, 145, 146),
    "magistral-medium-2506": (149, 134, 119, 99, 140, 128, 130, 116),
    "mistral-large-2411": (150, 151, 147, 143, 142, 144, 141, 152),
}

# Model name mappings: Arena ID -> Our cache name patterns
MODEL_NAME_MAPPINGS = {
    # Gemini models
    "gemini-3-pro": ["Gemini 3 Pro Preview", "Gemini 3 Pro"],
    "gemini-2.5-pro": ["Gemini 2.5 Pro"],
    "gemini-2.5-flash": ["Gemini 2.5 Flash"],
    "gemini-2.5-flash-preview-09-2025": ["Gemini 2.5 Flash Preview"],
    "gemini-2.0-flash-001": ["Gemini 2.0 Flash"],
    "gemini-1.5-pro-002": ["Gemini 1.5 Pro"],
    "gemini-1.5-flash-002": ["Gemini 1.5 Flash"],
    
    # Claude models
    "claude-opus-4-5-20251101": ["Claude Opus 4.5"],
    "claude-opus-4-5-20251101-thinking-32k": ["Claude Opus 4.5 (Reasoning)"],
    "claude-sonnet-4-5-20250929": ["Claude 4.5 Sonnet"],
    "claude-sonnet-4-5-20250929-thinking-32k": ["Claude 4.5 Sonnet (Reasoning)"],
    "claude-haiku-4-5-20251001": ["Claude 4.5 Haiku", "Claude 4.5 Haiku (Reasoning)"],
    "claude-opus-4-1-20250805": ["Claude 4.1 Opus", "Claude Opus 4"],
    "claude-opus-4-1-20250805-thinking-16k": ["Claude Opus 4 (Reasoning)"],
    "claude-sonnet-4-20250514": ["Claude 4 Sonnet"],
    "claude-sonnet-4-20250514-thinking-32k": ["Claude 4 Sonnet (Reasoning)"],
    "claude-3-5-sonnet-20241022": ["Claude 3.5 Sonnet"],
    "claude-3-5-sonnet-20240620": ["Claude 3.5 Sonnet v1"],
    "claude-3-5-haiku-20241022": ["Claude 3.5 Haiku"],
    "claude-3-7-sonnet-20250219": ["Claude 3.7 Sonnet"],
    "claude-3-7-sonnet-20250219-thinking-32k": ["Claude 3.7 Sonnet (Reasoning)"],
    "claude-3-opus-20240229": ["Claude 3 Opus"],
    
    # GPT models
    "gpt-5.1-high": ["GPT-5.1 (high)", "GPT-5.1"],
    "gpt-5-high": ["GPT-5 (high)", "GPT-5"],
    "gpt-5-chat": ["GPT-5"],
    "gpt-4.5-preview-2025-02-27": ["GPT-4.5 Preview"],
    "gpt-4.1-2025-04-14": ["GPT-4.1"],
    "gpt-4.1-mini-2025-04-14": ["GPT-4.1 mini"],
    "gpt-4.1-nano-2025-04-14": ["GPT-4.1 nano"],
    "chatgpt-4o-latest-20250326": ["ChatGPT-4o"],
    "gpt-4o-2024-05-13": ["GPT-4o"],
    "gpt-4o-2024-08-06": ["GPT-4o (Aug 2024)"],
    "gpt-4o-mini-2024-07-18": ["GPT-4o mini"],
    "gpt-4-turbo-2024-04-09": ["GPT-4 Turbo"],
    "gpt-oss-120b": ["gpt-oss-120B (high)"],
    "gpt-oss-20b": ["gpt-oss-20B (high)"],
    
    # OpenAI o-series
    "o3-2025-04-16": ["o3"],
    "o3-mini": ["o3-mini"],
    "o3-mini-high": ["o3-mini (high)"],
    "o4-mini-2025-04-16": ["o4-mini (high)"],
    "o1-2024-12-17": ["o1"],
    "o1-preview": ["o1-preview"],
    "o1-mini": ["o1-mini"],
    
    # Grok models
    "grok-4.1": ["Grok 4.1"],
    "grok-4.1-thinking": ["Grok 4.1 (Reasoning)"],
    "grok-4-0709": ["Grok 4"],
    "grok-3-preview-02-24": ["Grok 3"],
    "grok-3-mini-high": ["Grok 3 mini Reasoning (high)"],
    "grok-3-mini-beta": ["Grok 3 mini"],
    "grok-2-2024-08-13": ["Grok 2"],
    
    # DeepSeek models
    "deepseek-v3.2": ["DeepSeek V3.2"],
    "deepseek-v3.2-thinking": ["DeepSeek V3.2 (Reasoning)"],
    "deepseek-v3.1": ["DeepSeek V3.1"],
    "deepseek-v3.1-thinking": ["DeepSeek V3.1 (Reasoning)"],
    "deepseek-v3.1-terminus": ["DeepSeek V3.1 Terminus"],
    "deepseek-v3.1-terminus-thinking": ["DeepSeek V3.1 Terminus (Reasoning)"],
    "deepseek-v3-0324": ["DeepSeek V3"],
    "deepseek-v3": ["DeepSeek V3"],
    "deepseek-r1-0528": ["DeepSeek R1 0528"],
    "deepseek-r1": ["DeepSeek R1"],
    "deepseek-v2.5": ["DeepSeek V2.5"],
    
    # Kimi models
    "kimi-k2-thinking-turbo": ["Kimi K2 Thinking"],
    "kimi-k2-0905-preview": ["Kimi K2"],
    
    # Llama models
    "llama-3.1-405b-instruct-bf16": ["Llama 3.1 Instruct 405B"],
    "llama-3.1-405b-instruct-fp8": ["Llama 3.1 Instruct 405B"],
    "llama-3.3-70b-instruct": ["Llama 3.3 Instruct 70B"],
    "llama-3.1-70b-instruct": ["Llama 3.1 Instruct 70B"],
    
    # Qwen models
    "qwen3-max-preview": ["Qwen 3 Max"],
    "qwen3-235b-a22b-instruct-2507": ["Qwen 3 235B"],
    "qwen2.5-max": ["Qwen 2.5 Max"],
    "qwen2.5-72b-instruct": ["Qwen 2.5 72B Instruct"],
    "qwq-32b": ["QwQ 32B"],
    
    # Mistral models
    "mistral-large-3": ["Mistral Large 3"],
    "mistral-large-2407": ["Mistral Large 2407"],
    "mistral-medium-2508": ["Mistral Medium"],
    "mistral-small-2506": ["Mistral Small 3.2"],
    
    # Gemma models
    "gemma-3-27b-it": ["Gemma 3 27B Instruct"],
    "gemma-3-12b-it": ["Gemma 3 12B Instruct"],
    "gemma-3-4b-it": ["Gemma 3 4B Instruct"],
    
    # Amazon Nova
    "amazon-nova-pro-v1.0": ["Nova Pro"],
    "amazon-nova-lite-v1.0": ["Nova Lite"],
    "amazon-nova-micro-v1.0": ["Nova Micro"],
    
    # MiniMax
    "minimax-m1": ["MiniMax-M1"],
    "minimax-m2": ["MiniMax-M2"],
}


def normalize_model_name(name: str) -> str:
    """Normalize model name for matching."""
    # Convert to lowercase and remove common variations
    name = name.lower()
    name = re.sub(r'[_\-\s]+', ' ', name)
    name = re.sub(r'\(.*?\)', '', name)  # Remove parenthetical info
    name = name.strip()
    return name


def find_matching_model(arena_id: str, cache_models: list) -> Optional[Dict]:
    """Find a model in the cache that matches the arena ID."""
    # Direct mapping
    if arena_id in MODEL_NAME_MAPPINGS:
        for cache_name in MODEL_NAME_MAPPINGS[arena_id]:
            for model in cache_models:
                if model.get('name') == cache_name:
                    return model
    
    # Fuzzy matching
    arena_normalized = normalize_model_name(arena_id)
    for model in cache_models:
        model_name = model.get('name', '')
        model_normalized = normalize_model_name(model_name)
        
        # Check for substring match
        if arena_normalized in model_normalized or model_normalized in arena_normalized:
            return model
    
    return None


def main():
    """Update models cache with Arena rankings."""
    cache_path = Path(__file__).parent.parent.parent / "data" / "models_cache.json"
    
    print("=" * 60)
    print("UPDATING MODELS CACHE WITH LMARENA RANKINGS")
    print("=" * 60)
    print(f"\nData source: https://lmarena.ai/leaderboard")
    print(f"Cache path: {cache_path}\n")
    
    # Load cache
    with open(cache_path) as f:
        data = json.load(f)
    
    models = data.get('models', data) if isinstance(data, dict) else data
    print(f"Loaded {len(models)} models from cache.\n")
    
    # Track updates
    updated = 0
    matched = []
    unmatched = []
    
    # Update models with rankings
    for arena_id, rankings in ARENA_RANKINGS.items():
        model = find_matching_model(arena_id, models)
        
        if model:
            overall, expert, hard, coding, math, creative, instruction, longer = rankings
            
            model['arena_rank_overall'] = overall
            model['arena_rank_expert'] = expert
            model['arena_rank_hard'] = hard
            model['arena_rank_coding'] = coding
            model['arena_rank_math'] = math
            model['arena_rank_creative'] = creative
            model['arena_rank_instruction'] = instruction
            model['arena_rank_longer'] = longer
            
            # Add ELO if available
            if arena_id in ARENA_ELO_SCORES:
                model['arena_elo_text'] = ARENA_ELO_SCORES[arena_id]
            
            updated += 1
            matched.append((arena_id, model['name']))
        else:
            unmatched.append(arena_id)
    
    # Save updated cache
    if isinstance(data, dict) and 'models' in data:
        data['models'] = models
    else:
        data = models
    
    with open(cache_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    # Report
    print(f"Updated {updated} models with Arena rankings.\n")
    
    print("Matched models:")
    for arena_id, cache_name in matched[:20]:
        print(f"  {arena_id} -> {cache_name}")
    if len(matched) > 20:
        print(f"  ... and {len(matched) - 20} more")
    
    print(f"\nUnmatched Arena models ({len(unmatched)}):")
    for arena_id in unmatched[:15]:
        print(f"  {arena_id}")
    if len(unmatched) > 15:
        print(f"  ... and {len(unmatched) - 15} more")
    
    print(f"\nSaved to {cache_path}")
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
