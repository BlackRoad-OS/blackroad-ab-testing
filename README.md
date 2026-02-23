# blackroad-ab-testing

[![CI](https://github.com/BlackRoad-OS/blackroad-ab-testing/actions/workflows/ci.yml/badge.svg)](https://github.com/BlackRoad-OS/blackroad-ab-testing/actions/workflows/ci.yml)

> Statistically rigorous A/B testing framework for BlackRoad OS — consistent hashing, chi-squared significance, SQLite persistence.

---

## Features

- **Consistent hashing** — users always land in the same variant across sessions
- **Chi-squared significance testing** — know when your results are statistically meaningful
- **Multi-variant support** — A/B/C/D experiments with arbitrary traffic splits
- **SQLite persistence** — zero-dependency storage, no external services required
- **CLI + Python API** — use from the terminal or import as a library
- **Duplicate-safe** — idempotent assignment and conversion recording

---

## Installation

```bash
git clone https://github.com/BlackRoad-OS/blackroad-ab-testing.git
cd blackroad-ab-testing
pip install -r requirements.txt
```

---

## CLI Usage

### Create an experiment

```bash
python -m src.ab_testing --db myexp.db create button-color \
  --desc "Test CTA button color" \
  --variants "control:50,treatment:50" \
  --metric conversion
```

### Start / stop an experiment

```bash
python -m src.ab_testing --db myexp.db start button-color
python -m src.ab_testing --db myexp.db stop button-color
```

### Assign a user to a variant

```bash
python -m src.ab_testing --db myexp.db assign button-color user-123
# User 'user-123' assigned to variant: treatment
```

### Record a conversion

```bash
python -m src.ab_testing --db myexp.db convert button-color user-123
python -m src.ab_testing --db myexp.db convert button-color user-123 --value 49.99
```

### View stats and significance

```bash
python -m src.ab_testing --db myexp.db stats button-color --confidence 0.95
```

```json
{
  "variant_stats": {
    "control":   { "assignments": 512, "conversions": 51,  "conversion_rate_pct": 9.96 },
    "treatment": { "assignments": 488, "conversions": 122, "conversion_rate_pct": 25.0 }
  },
  "significance": {
    "chi2_stat": 54.21,
    "p_value": 0.0001,
    "is_significant": true,
    "recommended_winner": "treatment",
    "reason": "Chi2=54.2100 > critical=3.841 at 95% confidence (p≈0.0001)"
  }
}
```

### Get winning variant

```bash
python -m src.ab_testing --db myexp.db winner button-color
# Winner: treatment
```

### Export full results

```bash
python -m src.ab_testing --db myexp.db export button-color > results.json
```

### List all experiments

```bash
python -m src.ab_testing --db myexp.db list
```

---

## Python API

```python
from src.ab_testing import ABTestingFramework, Experiment, Variant

fw = ABTestingFramework(db_path="myexp.db")

# Create experiment
exp = Experiment(
    name="checkout-flow",
    variants=[
        Variant("control",   traffic_pct=50.0, description="Current flow"),
        Variant("treatment", traffic_pct=50.0, description="Simplified flow"),
    ],
    description="Test simplified checkout",
    hypothesis="Simplified flow increases conversions by 15%",
    metric="purchase",
)
fw.create_experiment(exp)
fw.start_experiment("checkout-flow")

# Assign users
variant = fw.assign_variant(exp, user_id="user-abc")  # deterministic

# Record conversions
fw.record_conversion("checkout-flow", user_id="user-abc", value=79.99)

# Check significance
sig = fw.calculate_significance(exp, confidence=0.95)
print(sig.is_significant, sig.recommended_winner, sig.reason)

# Get winner (also persists to DB)
winner = fw.get_winning_variant("checkout-flow")

# Export
results = fw.export_results("checkout-flow")
```

---

## Statistical Methodology

### Chi-Squared Test of Independence

The framework tests whether the difference in conversion rates between variants is statistically significant using the **chi-squared (χ²) test of independence**.

Given an experiment with variants V₁, V₂, …, Vₙ, a 2×n contingency table is built:

|          | Converted | Not converted |
|----------|-----------|---------------|
| Control  | c₁        | n₁ − c₁       |
| Treatment| c₂        | n₂ − c₂       |

The chi-squared statistic is:

```
χ² = Σ (observed − expected)² / expected
```

where expected frequencies are computed from row and column marginals.

The result is compared against the critical value for `(confidence, degrees_of_freedom = n_variants − 1)`.

**Confidence levels supported:** 90%, 95%, 99%, 99.9%

### Consistent Hashing

User-to-variant assignment uses SHA-256 hashing:

```python
hash_val = SHA256(f"{experiment_name}:{user_id}")
bucket   = (hash_val % 10_000) / 100.0   # → [0.00, 99.99]
```

Variants are assigned by cumulative traffic percentage buckets. This guarantees:
- The same user always sees the same variant
- Traffic splits are uniform across the hash space
- Changing experiment name produces independent assignments

---

## SQLite Schema

```sql
-- Experiment definitions
CREATE TABLE experiments (
    name         TEXT PRIMARY KEY,
    description  TEXT,
    hypothesis   TEXT,
    metric       TEXT,
    status       TEXT,   -- draft | running | stopped
    variants_json TEXT,
    created_at   TEXT,
    started_at   TEXT,
    ended_at     TEXT,
    winner       TEXT
);

-- User → variant assignments (one per user per experiment)
CREATE TABLE assignments (
    id              TEXT PRIMARY KEY,
    experiment_name TEXT,
    user_id         TEXT,
    variant_name    TEXT,
    assigned_at     TEXT,
    UNIQUE(experiment_name, user_id)
);

-- Conversion events (one per user per experiment)
CREATE TABLE conversions (
    id              TEXT PRIMARY KEY,
    experiment_name TEXT,
    user_id         TEXT,
    variant_name    TEXT,
    converted_at    TEXT,
    value           REAL,
    metadata        TEXT
);
```

---

## Running Tests

```bash
pytest tests/ -v --tb=short
```

---

## License

Proprietary — © BlackRoad OS, Inc. All rights reserved.
