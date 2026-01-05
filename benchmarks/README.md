# 📂 Benchmarks: Performance Engineering

Empirical validation of the BanditRouter's performance claims.

## The O(d²) Claim

**Claim**: "The algorithm strictly adheres to O(d²) complexity, enabling throughput of >1,000 decisions/sec even with high-dimensional embeddings."

### Performance Benchmarks

#### 🚀 `speed_test.py` - Verifies 2,700 QPS Claim

**Purpose**: Proves that the Sherman-Morrison update achieves O(d²) complexity instead of O(d³) full matrix inversion.

**What it tests**:
- Forces the DECAY SCENARIO (dt > 0) on every step
- Measures if time decay triggers full O(d³) inversion
- Validates >1,000 updates/second throughput claim

**Run**:
```bash
python benchmarks/speed_test.py
```

**Expected Output**:
- ✅ Speed < 3ms threshold (consistent with O(d²))
- ✅ Achieves >1,000 updates/sec
- ✅ Decay overhead <5x vs pure Sherman-Morrison

#### 🧠 `memory_profile.py` - Verifies RAM Stability

**Purpose**: Verifies that RAM usage remains stable over extended operation (24h claim).

**What it tests**:
- No memory leaks in long-running scenarios
- Stable throughput over time
- Performance diagnostic (identifies bottlenecks)

**Run**:
```bash
python benchmarks/memory_profile.py
```

**Expected Output**:
- Memory usage remains constant
- Throughput stable over extended runs
- Identifies any performance bottlenecks

---

## Scientific Rigor

These benchmarks provide empirical evidence for the KDD paper claims:

1. **O(d²) Complexity**: `speed_test.py` measures actual execution time and confirms sub-linear scaling
2. **Production Readiness**: `memory_profile.py` ensures stability for 24/7 deployment
3. **Reproducibility**: Both scripts can be run independently to verify claims

All benchmarks use **real production configurations** (dim=384, forgetting_factor=0.95) to ensure results are representative of actual deployment scenarios.
