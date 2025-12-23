# Table 3: Performance Dynamics on Challenging Prompts

| Scenario | System | Selected Model | Accuracy | Cost ($/1M) |
| :--- | :--- | :--- | :--- | :--- |
| Simple Query (Easy) | **BanditGPT (Ours)** | `amazon/nova-micro-v1` | 96.0% | $0.061 |
| Simple Query (Easy) | **LiteLLM (Static)** | `google/gemini-2.0-flash-001` | 96.0% | $0.175 |
| Simple Query (Easy) | **RouteLLM (Classifier)** | `openai/gpt-4o` | 96.0% | $4.375 |
| Simple Query (Easy) | **FrugalGPT (Cascade)** | `google/gemini-2.0-flash-001` | 96.0% | $0.175 |
| Simple Query (Easy) | **Oracle (Upper Bound)** | `anthropic/claude-3.5-haiku` | 96.0% | $1.600 |
| Standard Logic (Mid) | **BanditGPT (Ours)** | `qwen/qwen3-0.6b-04-28` | 75.0% | $0.398 |
| Standard Logic (Mid) | **LiteLLM (Static)** | `google/gemini-2.0-flash-001` | 93.0% | $0.175 |
| Standard Logic (Mid) | **RouteLLM (Classifier)** | `openai/gpt-4o` | 75.9% | $4.375 |
| Standard Logic (Mid) | **FrugalGPT (Cascade)** | `google/gemini-2.0-flash-001` | 93.0% | $0.175 |
| Standard Logic (Mid) | **Oracle (Upper Bound)** | `openai/o3-mini-high` | 95.0% | $1.925 |
| Complex Reasoning (Hard) | **BanditGPT (Ours)** | `amazon/nova-micro-v1` | 4.7% | $0.061 |
| Complex Reasoning (Hard) | **LiteLLM (Static)** | `google/gemini-2.0-flash-001` | 5.3% | $0.175 |
| Complex Reasoning (Hard) | **RouteLLM (Classifier)** | `google/gemini-2.0-flash-001` | 5.3% | $0.175 |
| Complex Reasoning (Hard) | **FrugalGPT (Cascade)** | `openai/gpt-4o` | 3.3% | $4.550 |
| Complex Reasoning (Hard) | **Oracle (Upper Bound)** | `google/gemini-3-pro-preview` | 37.2% | $4.500 |


*Note: 'Legacy Router' represents static routing rules (e.g. LiteLLM) which default to cost-efficient models regardless of complexity.*