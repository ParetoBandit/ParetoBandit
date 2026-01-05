# 🔄 User Workflow: Dynamic Model Registration

This guide explains how to safely add models at runtime while keeping your system stable.

---

## 📋 The Three-Step Workflow

### **Step 1: Register Dynamic Model**

Add a model at runtime - it exists in RAM immediately:

```python
# New model just released!
router.register_model("deepseek-v2", capabilities=["coding"], speed="fast")
```

**Status**: ✅ Model active  
**Location**: RAM only  
**Risk**: ⚠️ Will vanish on restart

---

### **Step 2: Verify It Works**

Send test requests and let the bandit learn:

```python
# Test with coding prompts
for prompt in test_prompts:
    model = router.select_arm(prompt)
    # ... get actual result ...
    router.update(prompt, model, reward)

# Check performance
stats = router.get_stats()
deepseek_stats = stats['arm_stats']['deepseek-v2']
print(f"Samples: {deepseek_stats['count']}, Avg: {deepseek_stats['reward']/deepseek_stats['count']:.2f}")
```

**Status**: 🧪 Testing in progress  
**Location**: RAM + learned weights  
**Risk**: ⚠️ Still temporary

---

### **Step 3: Commit Changes**

Happy with the results? Make it permanent:

```python
# Looks good! Save permanently
router.persist_to_config()
```

**What happens**:
1. Reads `config/models.json`
2. Adds `deepseek-v2` with learned priors
3. Atomic write (crash-safe)
4. Model now in config

**Status**: ✅ Permanent  
**Location**: RAM + Disk (`models.json`)  
**Risk**: ✅ Survives restarts

---

### **Step 4: Restart Safety**

Server restarts automatically load from config:

```python
# Server restarts
router = BanditGPT()
# deepseek-v2 is automatically loaded from models.json ✓
```

---

## 🏗️ Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     BANDIT ROUTER STORAGE                       │
└─────────────────────────────────────────────────────────────────┘

                    ┌─────────────────┐
                    │  User Requests  │
                    └────────┬────────┘
                             │
                             ▼
         ┌───────────────────────────────────────┐
         │   RAM (Hot Path - Millisecond Speed)  │
         │  ────────────────────────────────────  │
         │   A, A_inv, b, theta                  │
         │   • select_arm() → O(d)               │
         │   • update() → O(d²)                  │
         └───────────┬───────────┬───────────────┘
                     │           │
          ┌──────────┘           └──────────┐
          │                                   │
          ▼                                   ▼
┌─────────────────────┐           ┌─────────────────────┐
│   CHECKPOINT PATH   │           │   CONFIG PATH       │
│   (Automatic)       │           │   (Explicit Only)   │
├─────────────────────┤           ├─────────────────────┤
│ save()              │           │ persist_to_config()│
│ • Every 1000 updates│           │ • User-triggered    │
│ • On shutdown       │           │ • After testing     │
└─────────┬───────────┘           └─────────┬───────────┘
          │                                   │
          ▼                                   ▼
┌─────────────────────┐           ┌─────────────────────┐
│ ~/.bandit_gpt/      │           │ config/models.json  │
│ checkpoints/        │           │                     │
│ router_state.pkl    │           │ Human-Readable JSON │
│                     │           │ Version Controlled  │
│ Binary (Fast)       │           │ CI/CD Friendly      │
│ Crash Recovery      │           │ Fleet Deployment    │
└─────────┬───────────┘           └─────────┬───────────┘
          │                                   │
          │      On BanditGPT() init         │
          └──────────────┬──────────────────┘
                         │
                         ▼
                ┌────────────────┐
                │   Auto-Resume  │
                │   & Load       │
                └────────────────┘
```

---

## 🔑 Key Principles

### **1. Fast Loop (Left)**
- **What**: RAM updates (`A`, `theta`)
- **When**: Every millisecond
- **Why**: Performance (O(d²) updates)

### **2. Safety Path (Bottom Left)**
- **What**: Binary checkpoint
- **When**: Automatic (every 1000 updates)
- **Why**: Crash recovery

### **3. Config Path (Right)**
- **What**: Human-readable config
- **When**: Explicit user action only
- **Why**: Controlled, reviewable changes

---

## 🎯 Decision Tree: Which Method to Use?

```
┌─────────────────────────────────────────┐
│ Do you need to save something?          │
└────────┬──────────────┬─────────────────┘
         │              │
    YES  │              │  NO → Do nothing
         │              │      (auto-handled)
         ▼              │
┌─────────────────────┐ │
│ What are you saving?│ │
└──┬──────────────┬───┘ │
   │              │     │
   ▼              ▼     │
Learned        New      │
State        Model      │
   │              │     │
   ▼              ▼     │
save()    persist_to_   │
(auto!)    config()     │
                        │
                        ▼
                Want to share
                knowledge?
                        │
                        ▼
                  export_priors()
```

---

## ✅ Best Practices

### **DO**:

✅ **Test before persisting**
```python
# Test new model for a day
router.register_model("new-model", ...)
# ... wait 24h, monitor stats ...
if stats_look_good:
    router.persist_to_config()
```

✅ **Use version control**
```bash
git diff config/models.json  # Review changes
git commit -m "Add deepseek-v2 model"
```

✅ **Document changes**
```python
# Add comment in commit message
router.persist_to_config()
# Git commit: "Add deepseek-v2: 95% success rate on coding tasks"
```

### **DON'T**:

❌ **Don't persist untested models**
```python
# BAD: Immediately persist
router.register_model("untested-model", ...)
router.persist_to_config()  # ← Too early!
```

❌ **Don't call persist on every update**
```python
# BAD: Expensive I/O
router.update(...)
router.persist_to_config()  # ← Way too often!
```

❌ **Don't manually edit checkpoint files**
```bash
# DON'T: Binary format, will corrupt
vim ~/.bandit_gpt/checkpoints/router_state.pkl  # ← NoNo!
```

---

## 🔄 Complete Example

```python
from bandit_gpt.core import BanditGPT

# === Startup ===
router = BanditGPT()
# Loads:
# 1. config/models.json → model registry
# 2. ~/.bandit_gpt/checkpoints/router_state.pkl → learned state (if exists)

# === New Model Release ===
print("New model available: deepseek-v3")
router.register_model(
    "deepseek-v3",
    capabilities=["coding", "math"],
    speed="fast",
    cost="cheap"
)
print("⚠️ Model registered in RAM only - testing phase")

# === Testing Phase (24 hours) ===
print("Testing for 24 hours...")
# ... production traffic happens ...
# ... router learns which prompts deepseek-v3 is good at ...

# === Review Stats ===
stats = router.get_stats()
dv3_stats = stats['arm_stats']['deepseek-v3']
avg_reward = dv3_stats['reward'] / max(dv3_stats['count'], 1)

if avg_reward > 0.85 and dv3_stats['count'] > 100:
    print(f"✅ deepseek-v3 looks good! (avg={avg_reward:.2f}, n={dv3_stats['count']})")
    print("Making it permanent...")
    
    # === Commit Changes ===
    router.persist_to_config()
    # Now deepseek-v3 is in models.json
    
    print("💾 Changes saved to config/models.json")
    print("🎉 deepseek-v3 will survive server restarts!")
else:
    print(f"❌ deepseek-v3 underperforming (avg={avg_reward:.2f})")
    print("Not persisting - will try again later")

# === Optional: Export for Fleet ===
if input("Export to other regions? (y/n): ").lower() == 'y':
    router.export_priors("production_priors_2024-01.json")
    print("📤 Priors exported for fleet deployment")
```

---

## 🎓 Summary

The three-tier system provides:

1. **⚡ Performance**: RAM updates (millisecond speed)
2. **🔒 Safety**: Auto-checkpoints (crash recovery)
3. **🤝 Flexibility**: Explicit persist (controlled changes)

**Result**: You can add models dynamically without destabilizing your deployment! 🎉
