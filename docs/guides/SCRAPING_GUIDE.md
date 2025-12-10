# Comprehensive Benchmark Scraping Guide

This guide explains how to use the comprehensive benchmark scraping system to collect LLM performance data from multiple authoritative sources.

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [Data Sources](#data-sources)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Output Format](#output-format)
6. [Troubleshooting](#troubleshooting)

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements_scrapers.txt

# Optional: Install chromedriver for web scraping
brew install chromedriver  # Mac
# OR
apt-get install chromium-chromedriver  # Linux

# Run the scraper
python scrape_comprehensive_benchmarks.py
```

**Output**: 3 files with comprehensive benchmark data for 100+ models
- `comprehensive_benchmarks.csv`
- `comprehensive_benchmarks.json`
- `models_cache.json` (legacy format)

---

## Data Sources

The scraper collects data from 5 authoritative sources:

### 1. **OpenRouter** (Primary Source for Names & Pricing)
- **URL**: https://openrouter.ai/models
- **Data**: Canonical model names, pricing, context lengths, capabilities
- **Method**: API + Web scraping
- **Why**: Most comprehensive model catalog, real-time pricing

### 2. **LMSYS Chatbot Arena** (Human Preferences)
- **URL**: https://chat.lmsys.org/
- **Data**: Elo ratings, MT-Bench scores, rankings
- **Method**: Web scraping (fallback to curated data)
- **Why**: Gold standard for human preference evaluation

### 3. **Artificial Analysis** (Performance Metrics)
- **URL**: https://artificialanalysis.ai/
- **Data**: Latency (TTFT), throughput (tokens/sec), quality index
- **Method**: Web scraping (fallback to curated data)
- **Why**: Independent performance benchmarking

### 4. **Official Sources** (Capability Benchmarks)
- **URLs**: OpenAI docs, Google AI, Anthropic, xAI, Cohere
- **Data**: MMLU, GPQA, MATH, IFEval, HumanEval scores
- **Method**: Curated from technical reports & model cards
- **Why**: Authoritative benchmark scores

### 5. **HuggingFace Leaderboard** (Open Source)
- **URL**: https://huggingface.co/spaces/open-llm-leaderboard
- **Data**: Open source model benchmarks
- **Method**: API (via existing ModelDataMerger)
- **Why**: Comprehensive open source model data

---

## Installation

### Core Dependencies

```bash
pip install -r requirements_scrapers.txt
```

This installs:
- `requests` - HTTP requests
- `beautifulsoup4` - HTML parsing
- `selenium` - JavaScript rendering
- `pandas` - Data processing

### Optional: Selenium + Chromedriver

For enhanced web scraping (recommended):

**macOS:**
```bash
brew install chromedriver
```

**Linux:**
```bash
apt-get install chromium-chromedriver
```

**Verification:**
```bash
chromedriver --version
```

If chromedriver isn't available, the scraper will:
- ✅ Still work using API endpoints
- ⚠️ Skip web-only metrics (latency, rankings)

---

## Usage

### Basic Usage

```python
from llm_jury.data.scrapers import ComprehensiveBenchmarkAggregator

# Initialize
aggregator = ComprehensiveBenchmarkAggregator()

# Collect data
df = aggregator.collect_all_data()

# Export
aggregator.export_to_csv(df, "benchmarks.csv")
aggregator.print_summary(df)
```

### Command Line

```bash
# Fresh scrape (ignores cache)
python scrape_comprehensive_benchmarks.py

# View output
head comprehensive_benchmarks.csv
```

### Using Individual Scrapers

```python
from llm_jury.data.scrapers import (
    OpenRouterWebScraper,
    ChatbotArenaScraper,
    ArtificialAnalysisScraper,
    OfficialSourcesScraper
)

# Scrape OpenRouter only
or_scraper = OpenRouterWebScraper(use_selenium=True)
models = or_scraper.scrape()

# Scrape Arena ratings only
arena_scraper = ChatbotArenaScraper()
ratings = arena_scraper.scrape()
```

---

## Output Format

### Comprehensive Benchmarks CSV

Columns (40+ fields):

**Identity:**
- `model_name` - Short name (e.g., "gpt-4o")
- `model_id` - Full ID (e.g., "openai/gpt-4o")
- `display_name` - Human-readable name
- `provider` - Company (openai, anthropic, google, etc.)

**Capability Benchmarks:**
- `mmlu_score` (0-100) - General knowledge
- `gpqa_score` (0-100) - Graduate-level reasoning
- `math_score` (0-100) - Mathematical reasoning
- `ifeval_score` (0-100) - Instruction following
- `humaneval_score` (0-100) - Code generation
- `tool_use_ability` (0-1.0) - Function calling

**Performance Metrics:**
- `latency_ms` - Time to first token
- `throughput_tps` - Tokens per second
- `quality_index` (0-100) - Overall quality

**Pricing:**
- `input_cost_per_m` - $ per 1M input tokens
- `output_cost_per_m` - $ per 1M output tokens
- `blended_cost` - Weighted average (75% input + 25% output)

**Specifications:**
- `context_length` - Max tokens
- `param_count_b` - Parameters in billions
- `supported_parameters` - API capabilities
- `input_modalities` - Supported inputs
- `output_modalities` - Supported outputs

**Rankings:**
- `arena_elo` - LMSYS Elo rating
- `mt_bench_score` (1-10) - Conversation quality
- `rank` - Top weekly ranking
- `quality_score` - Composite score

**Metadata:**
- `source` - Data source(s)
- `series` - Model family
- `category` - Use case category

### Models Cache JSON (Legacy)

Compatible with existing code:

```json
{
  "metadata": {
    "generated_at": "2025-01-29T...",
    "total_models": 150
  },
  "models": [
    {
      "name": "gpt-4o",
      "mmlu_score": 88.7,
      "gpqa_score": 53.6,
      "input_cost_per_m": 2.5,
      ...
    }
  ]
}
```

---

## Troubleshooting

### Issue: Selenium/Chromedriver Not Found

**Error:**
```
selenium.common.exceptions.WebDriverException: chromedriver not found
```

**Solution:**
```bash
# Install chromedriver
brew install chromedriver  # Mac
apt-get install chromium-chromedriver  # Linux

# Or disable Selenium
or_scraper = OpenRouterWebScraper(use_selenium=False)
```

### Issue: Rate Limiting / 429 Errors

**Error:**
```
requests.exceptions.HTTPError: 429 Too Many Requests
```

**Solution:**
```python
# Increase rate limit delay
scraper = BaseScraper(rate_limit_delay=2.0)  # 2 seconds between requests
```

### Issue: No Data Collected

**Symptoms:**
```
❌ No data collected. Check errors above.
```

**Debugging:**
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run again to see detailed errors
aggregator.collect_all_data()
```

### Issue: Stale Cache

**Solution:**
```python
# Force fresh scrape
df = aggregator.collect_all_data(use_cache=False)
```

---

## Advanced Configuration

### Custom Cache Location

```python
aggregator = ComprehensiveBenchmarkAggregator(
    cache_dir="/custom/path/to/cache"
)
```

### Scraper Options

```python
# Disable Selenium
or_scraper = OpenRouterWebScraper(use_selenium=False)

# Custom rate limiting
scraper = BaseScraper(rate_limit_delay=3.0)  # 3 seconds

# Custom timeout
scraper.session.timeout = 30  # 30 seconds
```

### Selective Scraping

```python
# Only scrape specific sources
aggregator.scrapers = {
    'openrouter': OpenRouterWebScraper(),
    'official': OfficialSourcesScraper(),
}

df = aggregator.collect_all_data()
```

---

## Data Quality

### Coverage (Expected)

- **Total Models**: 150-200
- **With Benchmarks**: 50-80 (official + manual)
- **With Pricing**: 150+ (OpenRouter)
- **With Latency**: 30-50 (Artificial Analysis)
- **With Elo Ratings**: 30-50 (Arena)

### Data Sources by Priority

1. **Pricing & Specs**: OpenRouter API (authoritative)
2. **Benchmarks**: Official sources > HuggingFace
3. **Performance**: Artificial Analysis > estimated
4. **Ratings**: LMSYS Arena (manual fallback)

### Refresh Frequency

- **Cache TTL**: 24 hours
- **Recommended**: Weekly refresh for pricing
- **Critical**: Monthly refresh for benchmarks

---

## Integration with Existing Code

The scraper outputs `models_cache.json` in the same format as the existing cache generator:

```python
# Existing code works unchanged
from llm_jury.optimization.chebyshev_scorer import ChebyshevScorer
import json

with open('models_cache.json', 'r') as f:
    cache = json.load(f)
    models = cache['models']

# Use in plots, analysis, etc.
```

---

## Contributing

### Adding a New Scraper

1. Create new scraper class:

```python
from .base_scraper import BaseScraper

class MyNewScraper(BaseScraper):
    def get_source_name(self) -> str:
        return "My Source"
    
    def scrape(self) -> List[Dict]:
        # Implementation
        pass
```

2. Register in aggregator:

```python
# In aggregate_scraper.py
self.scrapers['mysource'] = MyNewScraper()
```

3. Add tests and documentation

---

## Support

**Issues**: https://github.com/your-repo/issues
**Documentation**: See `docs/` folder
**Questions**: Open a discussion

---

**Last Updated**: January 2025
**Version**: 1.0.0

