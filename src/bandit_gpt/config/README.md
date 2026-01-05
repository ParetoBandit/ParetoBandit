# Configuration Directory

This directory contains **immutable configuration data** that ships with the pip package.

## Contents

### `models.json`

**Purpose**: The source of truth for production model registry.

**Structure**:
```json
{
  "gpt-4": {
    "provider": "openai",
    "cost_per_1k_tokens": 0.03,
    "latency_ms": 2000,
    "capabilities": ["coding", "math", "reasoning"],
    ...
  },
  ...
}
```

**Important**:
- ✅ **Ships with pip**: End users get this file
- 🔒 **Immutable in production**: Changes via git/code only
- 📦 **Version controlled**: Model definitions tied to package version

## Why Separate from Code?

Following best practices:
- **Code** (`router.py`, `core.py`) = Algorithm logic
- **Config** (`config/`) = Data definitions
- **Assets** (`assets/`) = Pre-trained artifacts

This separation allows:
1. **Easy model updates**: Change `models.json` without touching core logic
2. **Version tracking**: Model registry evolves with package versions
3. **Clear responsibility**: Config = "what models exist", Code = "how to route"

## Adding New Models

To add a new model to the registry:

1. Edit `config/models.json`
2. Add model definition with required fields
3. Commit to version control
4. Deploy new package version

**DO NOT** allow the router to modify this file at runtime - it should be read-only in production.
