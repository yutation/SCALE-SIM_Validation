# GEMM Performance Anomaly Detection Report

## Overview
This report summarizes the detected anomalies (significant increases/decreases) in actual_duration measurements across different GEMM configurations.

**Total Anomalies Detected: 140**

## Detection Methodology

The anomaly detection uses multiple criteria:
1. **Absolute difference threshold**: Based on 95th percentile of changes within each (K,N) group
2. **Percentage change threshold**: Based on 95th percentile of percentage changes
3. **Z-score detection**: Changes exceeding mean + 2.5 standard deviations

An anomaly is flagged if it meets ANY of these criteria.

---

## Summary by Configuration

### K=128, N=128
- **Anomalies detected**: 34
- **Duration range**: 1.12 - 2.55 μs
- **Largest jump**: 0.154 μs at M=978 (+6.46%)
- **Thresholds**: Absolute > 0.098 μs, Percentage > 5.39%

**Key Observations**:
- Most anomalies are moderate (5-8% changes)
- Alternating increase/decrease patterns at several M values
- Notable spikes at M=122, 130, 210, 274, 386, 394

### K=128, N=1024
- **Anomalies detected**: 39
- **Duration range**: 1.71 - 9.77 μs
- **Largest jump**: 0.977 μs at M=1010 (+11.11%)
- **Thresholds**: Absolute > 0.156 μs, Percentage > 5.26%

**Key Observations**:
- Larger absolute changes compared to (128,128)
- Significant jumps in the range M=210-450
- Multiple anomalies with 7-9% changes
- Largest jump occurs at M=1010 (near the end of the range)

### K=1024, N=128
- **Anomalies detected**: 31
- **Duration range**: 2.14 - 8.41 μs
- **Thresholds**: Absolute > 0.109 μs, Percentage > 2.59%

**Key Observations**:
- **MASSIVE SPIKE at M=122**: +23.71% (from 2.39 to 2.95 μs)
- **HUGE JUMP at M=754**: +17.39% (from 6.07 to 7.12 μs)
- These are the most significant percentage changes in the entire dataset
- Otherwise relatively smooth progression

### K=1024, N=1024
- **Anomalies detected**: 36
- **Duration range**: 4.93 - 43.87 μs
- **Thresholds**: Absolute > 0.332 μs, Percentage > 2.78%

**Key Observations**:
- **EXTREME ANOMALIES** - Multiple massive jumps:
  - M=106: +24.64% (6.36 → 7.93 μs)
  - M=338: +43.50% (13.94 → 20.00 μs) ⚠️
  - M=362: +40.12% (14.61 → 20.47 μs) ⚠️
  - M=802: +28.03% (25.79 → 33.03 μs)
  - M=898: +26.76% (26.97 → 34.18 μs)
  - M=930: +26.85% (27.36 → 34.71 μs)
  - M=994: +25.01% (28.20 → 35.26 μs)
  - **M=1018: +53.07% (28.66 → 43.87 μs)** ⚠️⚠️⚠️ LARGEST ANOMALY
- Pattern of alternating spikes and drops in the range M=800-1024
- Largest absolute jump: 15.21 μs at M=1018

---

## Critical Anomalies (>20% change)

| K    | N    | M    | Change Type | Previous (μs) | Current (μs) | Change (%) |
|------|------|------|-------------|---------------|--------------|------------|
| 1024 | 128  | 122  | INCREASE    | 2.39          | 2.95         | +23.71%    |
| 1024 | 1024 | 106  | INCREASE    | 6.36          | 7.93         | +24.64%    |
| 1024 | 1024 | 338  | INCREASE    | 13.94         | 20.00        | +43.50%    |
| 1024 | 1024 | 354  | DECREASE    | 20.06         | 14.55        | -27.47%    |
| 1024 | 1024 | 362  | INCREASE    | 14.61         | 20.47        | +40.12%    |
| 1024 | 1024 | 802  | INCREASE    | 25.79         | 33.03        | +28.03%    |
| 1024 | 1024 | 898  | INCREASE    | 26.97         | 34.18        | +26.76%    |
| 1024 | 1024 | 930  | INCREASE    | 27.36         | 34.71        | +26.85%    |
| 1024 | 1024 | 994  | INCREASE    | 28.20         | 35.26        | +25.01%    |
| 1024 | 1024 | 1018 | INCREASE    | 28.66         | 43.87        | +53.07%    |

---

## Patterns and Insights

### 1. Configuration Dependency
- **Small matrices (K=128, N=128)**: Relatively stable with minor fluctuations
- **Large matrices (K=1024, N=1024)**: Highly unstable with extreme spikes

### 2. M-value Hotspots
Certain M values show consistent anomalies across configurations:
- M=106, 122, 338, 362 appear in multiple configurations
- Large M values (>800) show increased instability in K=1024, N=1024

### 3. Possible Causes
The anomalies could be due to:
- **Hardware effects**: Cache l