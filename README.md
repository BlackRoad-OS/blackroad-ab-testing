# blackroad-ab-testing

[![CI](https://github.com/BlackRoad-OS/blackroad-ab-testing/actions/workflows/ci.yml/badge.svg)](https://github.com/BlackRoad-OS/blackroad-ab-testing/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/blackroad-ab-testing.svg)](https://pypi.org/project/blackroad-ab-testing/)
[![Python](https://img.shields.io/pypi/pyversions/blackroad-ab-testing.svg)](https://pypi.org/project/blackroad-ab-testing/)
[![License](https://img.shields.io/badge/license-Proprietary-red.svg)](./LICENSE)

> **Statistically rigorous A/B testing for production.** Consistent hashing, chi-squared significance, SQLite persistence — zero external dependencies. Drop it into any Python backend or Stripe webhook handler in minutes.

---

## Table of Contents

1. [Features](#features)
2. [Installation](#installation)
   - [pip (PyPI)](#pip-pypi)
   - [npm wrapper](#npm-wrapper)
   - [From source](#from-source)
3. [Quick Start](#quick-start)
4. [CLI Usage](#cli-usage)
   - [Create an experiment](#create-an-experiment)
   - [Start / Stop](#start--stop-an-experiment)
   - [Assign a user](#assign-a-user-to-a-variant)
   - [Record a conversion](#record-a-conversion)
   - [View stats](#view-stats-and-significance)
   - [Get winner](#get-winning-variant)
   - [Export results](#export-full-results)
   - [List experiments](#list-all-experiments)
5. [Python API](#python-api)
6. [Stripe Integration](#stripe-integration)
7. [Statistical Methodology](#statistical-methodology)
   - [Chi-Squared Test](#chi-squared-test-of-independence)
   - [Consistent Hashing](#consistent-hashing)
8. [SQLite Schema](#sqlite-schema)
9. [End-to-End Testing](#end-to-end-testing)
10. [Running Tests](#running-tests)
11. [License](#license)

---

## Features

- **Consistent hashing** — users always land in the same variant across sessions
- **Chi-squared significance testing** — know when your results are statistically meaningful
- **Multi-variant support** — A/B/C/D experiments with arbitrary traffic splits
- **SQLite persistence** — zero-dependency storage, no external services required
- **CLI + Python API** — use from the terminal or import as a library
- **Duplicate-safe** — idempotent assignment and conversion recording
- **Stripe-ready** — built-in patterns for recording payment events as conversions

---

## Installation

### pip (PyPI)

```bash
pip install blackroad-ab-testing
```

### npm wrapper

A lightweight JavaScript/Node.js wrapper that shells out to the Python CLI is available for teams running Node backends:

```bash
npm install blackroad-ab-testing
```

```js
const { ABTest } = require('blackroad-ab-testing');

const test = new ABTest({ db: 'myexp.db' });
const variant = await test.assign('checkout-flow', req.user.id);
```

> **Note:** The npm package requires Python ≥ 3.9 to be available on the host.

### From source

```bash
git clone https://github.com/BlackRoad-OS/blackroad-ab-testing.git
cd blackroad-ab-testing
pip install -r requirements.txt
```

---

## Quick Start

```python
from src.ab_testing import ABTestingFramework, Experiment, Variant

fw = ABTestingFramework(db_path="experiments.db")

exp = Experiment(
    name="pricing-page",
    variants=[
        Variant("control",   traffic_pct=50.0, description="Current pricing"),
        Variant("treatment", traffic_pct=50.0, description="New pricing layout"),
    ],
    description="Test new pricing page layout",
    hypothesis="New layout increases plan upgrades by 20%",
    metric="upgrade",
)

fw.create_experiment(exp)
fw.start_experiment("pricing-page")

# Deterministic — same user always gets the same variant
variant = fw.assign_variant(exp, user_id="user-abc")
print(f"User sees: {variant}")

# Record purchase
fw.record_conversion("pricing-page", user_id="user-abc", value=99.00)

# Check results
sig = fw.calculate_significance(exp, confidence=0.95)
print(sig.is_significant, sig.recommended_winner)
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

# Assign users (deterministic across sessions)
variant = fw.assign_variant(exp, user_id="user-abc")

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

## Stripe Integration

Use this framework to run statistically rigorous A/B tests on your Stripe payment flows — pricing pages, checkout UX, plan variants, coupon offers, and more.

### Pattern: test a Stripe Checkout price

```python
import stripe
from src.ab_testing import ABTestingFramework, Experiment, Variant

stripe.api_key = "sk_live_..."
fw = ABTestingFramework(db_path="stripe_experiments.db")

# Define the experiment once
exp = Experiment(
    name="annual-vs-monthly",
    variants=[
        Variant("monthly", traffic_pct=50.0, description="Monthly plan — $29/mo"),
        Variant("annual",  traffic_pct=50.0, description="Annual plan — $249/yr"),
    ],
    description="Test annual vs monthly plan framing on pricing page",
    hypothesis="Annual framing increases LTV by 30%",
    metric="stripe_checkout_completed",
)
fw.create_experiment(exp)
fw.start_experiment("annual-vs-monthly")

# In your request handler — assign a variant deterministically by customer ID
def get_pricing_page(customer_id: str):
    variant = fw.assign_variant(exp, user_id=customer_id)
    price_id = (
        "price_annual_249"   if variant == "annual"
        else "price_monthly_29"
    )
    session = stripe.checkout.Session.create(
        customer=customer_id,
        line_items=[{"price": price_id, "quantity": 1}],
        mode="subscription",
        success_url="https://app.example.com/welcome",
        cancel_url="https://app.example.com/pricing",
    )
    return session.url
```

### Pattern: record conversion from Stripe webhook

```python
import stripe
from flask import Flask, request
from src.ab_testing import ABTestingFramework

app = Flask(__name__)
fw  = ABTestingFramework(db_path="stripe_experiments.db")

@app.route("/stripe/webhook", methods=["POST"])
def stripe_webhook():
    payload = request.data
    sig_header = request.headers.get("Stripe-Signature")
    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, "whsec_..."
        )
    except stripe.error.SignatureVerificationError:
        return "Bad signature", 400

    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
        customer_id = session.get("customer")
        amount_paid = session.get("amount_total", 0) / 100  # cents → dollars

        if customer_id:
            try:
                fw.record_conversion(
                    "annual-vs-monthly",
                    user_id=customer_id,
                    value=amount_paid,
                    metadata={"stripe_session_id": session["id"]},
                )
            except ValueError:
                pass  # user not in this experiment

    return "", 200
```

### Analyzing Stripe experiment results

```python
sig = fw.calculate_significance(exp, confidence=0.95)

print(f"Significant: {sig.is_significant}")
print(f"Recommended plan framing: {sig.recommended_winner}")
for variant, stats in sig.variant_stats.items():
    print(
        f"  {variant}: {stats['assignments']} users, "
        f"{stats['conversion_rate_pct']}% converted, "
        f"${stats['total_value']:.2f} total revenue"
    )
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
    name          TEXT PRIMARY KEY,
    description   TEXT,
    hypothesis    TEXT,
    metric        TEXT,
    status        TEXT,    -- draft | running | stopped
    variants_json TEXT,
    created_at    TEXT,
    started_at    TEXT,
    ended_at      TEXT,
    winner        TEXT
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

-- Indexes for high-throughput production workloads
CREATE INDEX idx_asgn_exp     ON assignments(experiment_name);
CREATE INDEX idx_asgn_user    ON assignments(user_id);
CREATE INDEX idx_conv_exp     ON conversions(experiment_name);
CREATE INDEX idx_conv_variant ON conversions(variant_name);
```

---

## End-to-End Testing

The following e2e flow exercises every stage of an experiment lifecycle — create, start, assign, convert, analyse, export.

```python
"""e2e_test.py — run with: python e2e_test.py"""
import os
import tempfile
from src.ab_testing import ABTestingFramework, Experiment, Variant

def run_e2e():
    with tempfile.TemporaryDirectory() as tmpdir:
        db = os.path.join(tmpdir, "e2e.db")
        fw = ABTestingFramework(db_path=db)

        # 1. Create & start
        exp = Experiment(
            name="e2e-checkout",
            variants=[
                Variant("control",   traffic_pct=50.0),
                Variant("treatment", traffic_pct=50.0),
            ],
            description="E2E test experiment",
            metric="purchase",
        )
        fw.create_experiment(exp)
        fw.start_experiment("e2e-checkout")
        assert fw.get_experiment("e2e-checkout").status == "running"

        # 2. Assign 500 users (each call is O(1) — one DB read + one write per new user)
        assignments = {}
        for i in range(500):
            assignments[f"u{i}"] = fw.assign_variant(exp, f"u{i}")
        assert all(v in ("control", "treatment") for v in assignments.values())

        # 3. Consistent re-assignment
        assert fw.assign_variant(exp, "u0") == assignments["u0"]

        # 4. Record conversions — heavy treatment lift
        for i in range(500):
            uid = f"u{i}"
            v = assignments[uid]
            if (v == "treatment" and i % 3 == 0) or (v == "control" and i % 10 == 0):
                fw.record_conversion("e2e-checkout", uid, value=49.99)

        # 5. Significance
        sig = fw.calculate_significance(exp, confidence=0.95)
        assert sig.is_significant, "Expected significant result with heavy treatment lift"
        assert sig.recommended_winner == "treatment"
        print(f"  chi2={sig.chi2_stat}, p={sig.p_value}, winner={sig.recommended_winner}")

        # 6. Winning variant persisted
        winner = fw.get_winning_variant("e2e-checkout")
        assert winner == "treatment"

        # 7. Export
        results = fw.export_results("e2e-checkout")
        assert results["significance"]["is_significant"] is True

        # 8. Stop
        fw.stop_experiment("e2e-checkout")
        assert fw.get_experiment("e2e-checkout").status == "stopped"

        print("✅  E2E test passed")

if __name__ == "__main__":
    run_e2e()
```

Run it:

```bash
python e2e_test.py
# ✅  E2E test passed
```

---

## Running Tests

### Unit & integration tests

```bash
pytest tests/ -v --tb=short
```

### Lint

```bash
flake8 src/ tests/ --max-line-length=120
```

### CI

All pushes and pull requests to `main` run the full test suite automatically via GitHub Actions. See [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

---

## License

Proprietary — © BlackRoad OS, Inc. All rights reserved.
