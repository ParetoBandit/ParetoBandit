# 💾 Complete Save/Persist API - Three Methods for Different Use Cases

BanditGPT now provides **three distinct save methods** for different production scenarios:

---

## 📋 The Three Save Methods

| Method | Use Case | What Gets Saved | Format | When to Use |
|--------|----------|----------------|--------|-------------|
| `save()` | Pause/Resume | Full mathematical state | Binary (.pkl) | Server restarts |
| `export_priors()` | Transfer Learning | Human-readable priors | JSON | Share knowledge |
| `persist_to_config()` | Dynamic Registration | Models + priors to config | JSON | Permanent changes |

---

## 1. `save()` - Pause/Resume Checkpoint

**Problem**: "I need to restart the server but don't want to lose learning"

```python
# Before shutdown
router.save()

# After restart - automatically resumes
router = BanditGPT()  # Loads checkpoint in __init__
```

**What Gets Saved**:
- Matrices: `A`, `A_inv`, `b`, `theta`
- Metadata: `t`, `last_update`, model registry
- Location: `~/.bandit_gpt/checkpoints/router_state.pkl`

**Safety**: Atomic writes (temp file + rename)

---

## 2. `export_priors()` - Transfer Learning

**Problem**: "Share learned knowledge across deployments"

```python
# After learning period
router.export_priors("my_learned_priors.json")

# Deploy to fleet
config = RouterConfig(intuition=load_priors("my_learned_priors.json"))
new_routers = [BanditGPT(config) for _ in range(100)]
```

**Output Format**:
```json
{
  "gpt-4o": {
    "bias": 1.2,
    "weights": {
      "has_latex": 0.8,
      "complexity_score": 1.1
    },
    "metadata": {
      "samples": 1250,
      "avg_reward": 0.89,
      "status": "active"
    }
  }
}
```

**Use Cases**:
- Fleet deployment with shared knowledge
- Debugging model performance
- Creating "golden" priors
- Inspecting what the bandit learned

---

## 3. `persist_to_config()` - Write-Back Sync (NEW!)

**Problem**: "I registered a new model dynamically - make it permanent"

```python
# Add models at runtime
router.register_model("deepseek-v3", capabilities=["coding"])
router.register_model("claude-opus-4", capabilities=["creative"])

# Test them out... happy with results?
router.persist_to_config()

# Now they survive server restarts!
```

**What It Does**:
1. Reads existing `models.json`
2. Merges with current registry
3. Adds learned priors for each model
4. Atomically writes back
5. Preserves existing metadata (description, cost, etc.)

**Safety Features**:
- ✅ Atomic write (temp file + rename)
- ✅ Preserves existing config structure
- ✅ Merges rather than overwrites
- ✅ No corruption on crash

**Output**: Updated `src/bandit_gpt/config/models.json`

---

## 🔄 When to Use Which Method

### **Scenario 1: Server Restart (Maintenance)**
```python
# Automatic - no action needed!
router.save()  # Auto-called every 1000 updates
# On restart:
router = BanditGPT()  # Auto-resumes
```

### **Scenario 2: Fleet Deployment**
```python
# Production router learns for 1 week
router.export_priors("golden_priors.json")

# Deploy to 100 servers
for server in fleet:
    config = RouterConfig(intuition=load("golden_priors.json"))
    server.router = BanditGPT(config)
```

### **Scenario 3: Adding New Model**
```python
# New model released
router.register_model("gpt-5", capabilities=["reasoning"])

# Test it for a day...
# Looks good? Make it permanent:
router.persist_to_config()

# Now gpt-5 is in models.json for next deployment
```

---

## 🎯 Complete Workflow Example

```python
from bandit_gpt.core import BanditGPT

# === DAY 1: Initial Deployment ===
router = BanditGPT()

# Register models (from config/models.json)
# Already loaded automatically

# === DAY 7: Add New Model ===
router.register_model("deepseek-v3", capabilities=["coding"], cost="cheap")

# Test with production traffic...
# Performance looks good!

# Make it permanent
router.persist_to_config()  # ← New model now in models.json

# === DAY 14: Export Knowledge ===
router.export_priors("week2_learnings.json")
# Share with:
# - Staging environment
# - Other regions
# - Development team for analysis

# === DAY 15: Server Restart ===
router.save()  # Explicit save before maintenance
# ... restart happens ...
router = BanditGPT()  # Auto-resumes from checkpoint
# deepseek-v3 is still there (from persist_to_config)
```

---

## 🔍 Comparison Table

| Aspect | `save()` | `export_priors()` | `persist_to_config()` |
|--------|----------|-------------------|----------------------|
| **Format** | Binary (pickle) | JSON (human) | JSON (config) |
| **Size** | Large (~MB) | Small (~KB) | Small (~KB) |
| **Speed** | Fast | Fast | Medium |
| **Auto-called** | Yes (every 1000 updates) | No | No |
| **Auto-loaded** | Yes (on init) | No | Yes (via config) |
| **Editable** | No | Yes | Yes |
| **Use in CI/CD** | No | Yes | Yes |
| **Survives uninstall** | Yes (~/.bandit_gpt) | Yes | Yes (in repo) |

---

## ⚠️ Best Practices

### **DO**:
✅ Use `save()` for routine checkpointing (auto-handled)  
✅ Use `export_priors()` for knowledge sharing  
✅ Use `persist_to_config()` after testing new models  
✅ Call `persist_to_config()` before major deployments  

### **DON'T**:
❌ Call `persist_to_config()` on every update (expensive I/O)  
❌ Manually edit `router_state.pkl` (binary format)  
❌ Deploy `models.json` changes without testing  

---

## 🎓 Summary

BanditGPT provides a **three-tier persistence system**:

1. **`save()`** → Automatic crash recovery (binary checkpoint)
2. **`export_priors()`** → Knowledge sharing (human-readable JSON)
3. **`persist_to_config()`** → Dynamic config updates (write-back sync)

**Result**: Production-ready state management that balances:
- ⚡ Performance (auto-checkpointing)
- 🔒 Safety (atomic writes)
- 🤝 Usability (explicit persist)
- 📊 Interpretability (JSON exports)

This solves the "Immutable Infrastructure vs Dynamic Usability" tension elegantly! 🎉
