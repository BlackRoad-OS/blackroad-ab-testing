#!/usr/bin/env python3
"""BlackRoad A/B Testing Framework - Statistically rigorous experiment management."""
from __future__ import annotations
import argparse, csv, hashlib, json, math, sqlite3, sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Chi-squared critical values for common confidence levels
# keys are (confidence_level, degrees_of_freedom)
CHI2_CRITICAL = {
    (0.90, 1): 2.706, (0.90, 2): 4.605, (0.90, 3): 6.251,
    (0.95, 1): 3.841, (0.95, 2): 5.991, (0.95, 3): 7.815,
    (0.99, 1): 6.635, (0.99, 2): 9.210, (0.99, 3): 11.345,
    (0.999, 1): 10.828, (0.999, 2): 13.816, (0.999, 3): 16.266,
}

VARIANT_COLORS = ["control", "treatment_a", "treatment_b", "treatment_c", "treatment_d"]


@dataclass
class Variant:
    name: str
    traffic_pct: float
    description: str = ""


@dataclass
class Experiment:
    name: str
    variants: List[Variant]
    description: str = ""
    hypothesis: str = ""
    metric: str = "conversion"
    status: str = "draft"
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    started_at: str = ""
    ended_at: str = ""
    winner: str = ""

    def __post_init__(self):
        total = sum(v.traffic_pct for v in self.variants)
        if not (99.9 <= total <= 100.1):
            raise ValueError(f"Variant traffic percentages must sum to 100, got {total}")
        names = [v.name for v in self.variants]
        if len(names) != len(set(names)):
            raise ValueError("Variant names must be unique")

    def variant_names(self) -> List[str]:
        return [v.name for v in self.variants]


@dataclass
class SignificanceResult:
    chi2_stat: float
    p_value: float
    is_significant: bool
    confidence: float
    degrees_of_freedom: int
    variant_stats: Dict[str, Dict]
    recommended_winner: str
    reason: str


class ABTestingFramework:
    """Full A/B testing framework with consistent hashing and chi-squared significance."""

    def __init__(self, db_path: str = "ab_testing.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS experiments (
                    name TEXT PRIMARY KEY,
                    description TEXT DEFAULT '',
                    hypothesis TEXT DEFAULT '',
                    metric TEXT DEFAULT 'conversion',
                    status TEXT DEFAULT 'draft',
                    variants_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    started_at TEXT DEFAULT '',
                    ended_at TEXT DEFAULT '',
                    winner TEXT DEFAULT ''
                );
                CREATE TABLE IF NOT EXISTS assignments (
                    id TEXT PRIMARY KEY,
                    experiment_name TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    variant_name TEXT NOT NULL,
                    assigned_at TEXT NOT NULL,
                    UNIQUE(experiment_name, user_id),
                    FOREIGN KEY (experiment_name) REFERENCES experiments(name)
                );
                CREATE TABLE IF NOT EXISTS conversions (
                    id TEXT PRIMARY KEY,
                    experiment_name TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    variant_name TEXT NOT NULL,
                    converted_at TEXT NOT NULL,
                    value REAL DEFAULT 1.0,
                    metadata TEXT DEFAULT '{}',
                    FOREIGN KEY (experiment_name) REFERENCES experiments(name)
                );
                CREATE INDEX IF NOT EXISTS idx_asgn_exp ON assignments(experiment_name);
                CREATE INDEX IF NOT EXISTS idx_asgn_user ON assignments(user_id);
                CREATE INDEX IF NOT EXISTS idx_conv_exp ON conversions(experiment_name);
                CREATE INDEX IF NOT EXISTS idx_conv_variant ON conversions(variant_name);
            """)

    def _gen_id(self, prefix: str = "id") -> str:
        ts = datetime.utcnow().isoformat()
        h = hashlib.sha256(ts.encode()).hexdigest()[:12]
        return f"{prefix}-{h}"

    def create_experiment(self, experiment: Experiment) -> Experiment:
        """Create and store a new experiment."""
        with sqlite3.connect(self.db_path) as conn:
            existing = conn.execute("SELECT name FROM experiments WHERE name=?", (experiment.name,)).fetchone()
            if existing:
                raise ValueError(f"Experiment '{experiment.name}' already exists")
            variants_json = json.dumps([asdict(v) for v in experiment.variants])
            conn.execute(
                "INSERT INTO experiments VALUES (?,?,?,?,?,?,?,?,?,?)",
                (experiment.name, experiment.description, experiment.hypothesis,
                 experiment.metric, experiment.status, variants_json,
                 experiment.created_at, experiment.started_at,
                 experiment.ended_at, experiment.winner)
            )
        return experiment

    def start_experiment(self, name: str) -> Experiment:
        """Start a draft experiment."""
        exp = self.get_experiment(name)
        if not exp:
            raise KeyError(f"Experiment not found: {name}")
        if exp.status != "draft":
            raise ValueError(f"Experiment '{name}' is already {exp.status}")
        now = datetime.utcnow().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE experiments SET status='running', started_at=? WHERE name=?", (now, name))
        exp.status = "running"
        exp.started_at = now
        return exp

    def stop_experiment(self, name: str) -> Experiment:
        """Stop a running experiment."""
        exp = self.get_experiment(name)
        if not exp:
            raise KeyError(f"Experiment not found: {name}")
        now = datetime.utcnow().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE experiments SET status='stopped', ended_at=? WHERE name=?", (now, name))
        exp.status = "stopped"
        exp.ended_at = now
        return exp

    def assign_variant(self, experiment: Experiment, user_id: str) -> str:
        """Assign a variant to a user using consistent hashing. Returns variant name."""
        # Check for existing assignment
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT variant_name FROM assignments WHERE experiment_name=? AND user_id=?",
                (experiment.name, user_id)
            ).fetchone()
            if row:
                return row[0]

        # Consistent hash: deterministic bucket assignment
        hash_input = f"{experiment.name}:{user_id}"
        hash_val = int(hashlib.sha256(hash_input.encode()).hexdigest(), 16)
        bucket = (hash_val % 10000) / 100.0  # 0.00 to 99.99

        # Assign based on cumulative traffic percentages
        cumulative = 0.0
        assigned = experiment.variants[0].name
        for variant in experiment.variants:
            cumulative += variant.traffic_pct
            if bucket < cumulative:
                assigned = variant.name
                break

        # Store assignment
        with sqlite3.connect(self.db_path) as conn:
            aid = self._gen_id("asgn")
            try:
                conn.execute(
                    "INSERT INTO assignments VALUES (?,?,?,?,?)",
                    (aid, experiment.name, user_id, assigned, datetime.utcnow().isoformat())
                )
            except sqlite3.IntegrityError:
                # Race condition: re-fetch
                row = conn.execute(
                    "SELECT variant_name FROM assignments WHERE experiment_name=? AND user_id=?",
                    (experiment.name, user_id)
                ).fetchone()
                return row[0] if row else assigned

        return assigned

    def record_conversion(self, experiment_name: str, user_id: str,
                          value: float = 1.0, metadata: Optional[Dict] = None) -> bool:
        """Record a conversion event for a user. Returns True if recorded, False if duplicate."""
        with sqlite3.connect(self.db_path) as conn:
            # Get variant assignment
            row = conn.execute(
                "SELECT variant_name FROM assignments WHERE experiment_name=? AND user_id=?",
                (experiment_name, user_id)
            ).fetchone()
            if not row:
                raise ValueError(f"User '{user_id}' has no assignment for experiment '{experiment_name}'")
            variant_name = row[0]

            # Check for duplicate conversion
            existing = conn.execute(
                "SELECT id FROM conversions WHERE experiment_name=? AND user_id=?",
                (experiment_name, user_id)
            ).fetchone()
            if existing:
                return False

            conv_id = self._gen_id("conv")
            conn.execute(
                "INSERT INTO conversions VALUES (?,?,?,?,?,?,?)",
                (conv_id, experiment_name, user_id, variant_name,
                 datetime.utcnow().isoformat(), value,
                 json.dumps(metadata or {}))
            )
        return True

    def get_variant_stats(self, experiment_name: str) -> Dict[str, Dict]:
        """Get assignment and conversion counts per variant."""
        with sqlite3.connect(self.db_path) as conn:
            asgn_rows = conn.execute(
                "SELECT variant_name, COUNT(*) as cnt FROM assignments WHERE experiment_name=? GROUP BY variant_name",
                (experiment_name,)
            ).fetchall()
            conv_rows = conn.execute(
                "SELECT variant_name, COUNT(*) as cnt, SUM(value) as total_value "
                "FROM conversions WHERE experiment_name=? GROUP BY variant_name",
                (experiment_name,)
            ).fetchall()

        assignments = {r[0]: r[1] for r in asgn_rows}
        conversions = {r[0]: (r[1], r[2] or 0.0) for r in conv_rows}

        stats = {}
        for variant_name, asgn_count in assignments.items():
            conv_count, conv_value = conversions.get(variant_name, (0, 0.0))
            rate = conv_count / asgn_count if asgn_count > 0 else 0.0
            stats[variant_name] = {
                "assignments": asgn_count,
                "conversions": conv_count,
                "conversion_rate": round(rate, 4),
                "conversion_rate_pct": round(rate * 100, 2),
                "total_value": round(conv_value, 2),
            }
        return stats

    def calculate_significance(self, experiment: Experiment,
                               confidence: float = 0.95) -> SignificanceResult:
        """Calculate chi-squared significance between all variant pairs."""
        stats = self.get_variant_stats(experiment.name)
        variant_names = list(stats.keys())

        if len(variant_names) < 2:
            return SignificanceResult(
                chi2_stat=0.0, p_value=1.0, is_significant=False,
                confidence=confidence, degrees_of_freedom=0,
                variant_stats=stats, recommended_winner="",
                reason="Insufficient variant data for significance test",
            )

        # Build observed frequency table: [assignments, non-conversions]
        observed = []
        for vname in variant_names:
            s = stats.get(vname, {"assignments": 0, "conversions": 0})
            n = s["assignments"]
            c = s["conversions"]
            observed.append([c, n - c])  # [converted, not_converted]

        dof = len(variant_names) - 1
        chi2 = self._chi2_stat(observed)
        p_value = self._p_value_approx(chi2, dof)

        critical = CHI2_CRITICAL.get((confidence, dof), CHI2_CRITICAL.get((confidence, 1), 3.841))
        is_sig = chi2 > critical

        # Find recommended winner (highest conversion rate with min 10 samples)
        winner = ""
        best_rate = -1.0
        for vname, s in stats.items():
            if s["assignments"] >= 10 and s["conversion_rate"] > best_rate:
                best_rate = s["conversion_rate"]
                winner = vname

        reason = (
            f"Chi2={chi2:.4f} {'>' if is_sig else '<='} critical={critical:.3f} at {confidence*100:.0f}% confidence"
            f" (p\u2248{p_value:.4f})"
        )

        return SignificanceResult(
            chi2_stat=round(chi2, 4), p_value=round(p_value, 4),
            is_significant=is_sig, confidence=confidence,
            degrees_of_freedom=dof, variant_stats=stats,
            recommended_winner=winner, reason=reason,
        )

    def _chi2_stat(self, observed: List[List[int]]) -> float:
        """Compute chi-squared statistic from observed frequency table."""
        row_totals = [sum(row) for row in observed]
        col_totals = [sum(row[j] for row in observed) for j in range(len(observed[0]))]
        grand_total = sum(row_totals)
        if grand_total == 0:
            return 0.0
        chi2 = 0.0
        for i, row in enumerate(observed):
            for j, obs in enumerate(row):
                expected = (row_totals[i] * col_totals[j]) / grand_total
                if expected > 0:
                    chi2 += (obs - expected) ** 2 / expected
        return chi2

    def _p_value_approx(self, chi2: float, dof: int) -> float:
        """Approximate p-value from chi-squared statistic using regularized incomplete gamma."""
        if chi2 <= 0 or dof <= 0:
            return 1.0
        k = dof / 2.0
        x = chi2 / 2.0
        try:
            p = 1.0 - self._regularized_gamma(k, x)
        except Exception:
            p = 1.0
        return max(0.0, min(1.0, p))

    def _regularized_gamma(self, a: float, x: float, iterations: int = 100) -> float:
        """Compute regularized lower incomplete gamma function P(a, x)."""
        if x < 0:
            return 0.0
        if x == 0:
            return 0.0
        term = 1.0 / a
        total = term
        for n in range(1, iterations):
            term *= x / (a + n)
            total += term
            if abs(term) < 1e-10:
                break
        return total * math.exp(-x + a * math.log(x) - math.lgamma(a))

    def get_winning_variant(self, experiment_name: str, confidence: float = 0.95) -> Optional[str]:
        """Return the winning variant if significant, else None."""
        exp = self.get_experiment(experiment_name)
        if not exp:
            raise KeyError(f"Experiment not found: {experiment_name}")
        result = self.calculate_significance(exp, confidence=confidence)
        if result.is_significant and result.recommended_winner:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("UPDATE experiments SET winner=? WHERE name=?",
                             (result.recommended_winner, experiment_name))
            return result.recommended_winner
        return None

    def export_results(self, experiment_name: str) -> Dict:
        """Export full experiment results as a dict."""
        exp = self.get_experiment(experiment_name)
        if not exp:
            raise KeyError(f"Experiment not found: {experiment_name}")
        stats = self.get_variant_stats(experiment_name)
        sig = self.calculate_significance(exp)
        return {
            "experiment": asdict(exp),
            "variant_stats": stats,
            "significance": {
                "chi2_stat": sig.chi2_stat,
                "p_value": sig.p_value,
                "is_significant": sig.is_significant,
                "confidence": sig.confidence,
                "recommended_winner": sig.recommended_winner,
                "reason": sig.reason,
            },
            "exported_at": datetime.utcnow().isoformat(),
        }

    def get_experiment(self, name: str) -> Optional[Experiment]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM experiments WHERE name=?", (name,)).fetchone()
            if not row:
                return None
            variants = [Variant(**v) for v in json.loads(row["variants_json"])]
            return Experiment(
                name=row["name"], variants=variants,
                description=row["description"], hypothesis=row["hypothesis"],
                metric=row["metric"], status=row["status"],
                created_at=row["created_at"], started_at=row["started_at"],
                ended_at=row["ended_at"], winner=row["winner"],
            )

    def list_experiments(self) -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT name,status,metric,created_at,winner FROM experiments ORDER BY created_at DESC"
            ).fetchall()
            return [dict(r) for r in rows]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ab-test", description="BlackRoad A/B Testing Framework")
    parser.add_argument("--db", default="ab_testing.db")
    sub = parser.add_subparsers(dest="command")

    create = sub.add_parser("create", help="Create experiment")
    create.add_argument("name")
    create.add_argument("--desc", default="")
    create.add_argument("--variants", default="control:50,treatment:50",
                        help="comma-separated name:pct pairs")
    create.add_argument("--metric", default="conversion")

    for cmd in ["start", "stop"]:
        p = sub.add_parser(cmd, help=f"{cmd.title()} experiment")
        p.add_argument("name")

    assign = sub.add_parser("assign", help="Assign user to variant")
    assign.add_argument("experiment")
    assign.add_argument("user_id")

    conv = sub.add_parser("convert", help="Record conversion")
    conv.add_argument("experiment")
    conv.add_argument("user_id")
    conv.add_argument("--value", type=float, default=1.0)

    stats = sub.add_parser("stats", help="Show variant stats")
    stats.add_argument("experiment")
    stats.add_argument("--confidence", type=float, default=0.95)

    winner = sub.add_parser("winner", help="Get winning variant")
    winner.add_argument("experiment")
    winner.add_argument("--confidence", type=float, default=0.95)

    export = sub.add_parser("export", help="Export results")
    export.add_argument("experiment")

    sub.add_parser("list", help="List experiments")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)
    fw = ABTestingFramework(db_path=args.db)

    if args.command == "create":
        variants = []
        for part in args.variants.split(","):
            name, pct = part.strip().split(":")
            variants.append(Variant(name=name.strip(), traffic_pct=float(pct.strip())))
        exp = Experiment(name=args.name, variants=variants, description=args.desc, metric=args.metric)
        fw.create_experiment(exp)
        print(f"Created experiment '{args.name}' with variants: {[v.name for v in variants]}")
    elif args.command == "start":
        fw.start_experiment(args.name)
        print(f"Started experiment '{args.name}'")
    elif args.command == "stop":
        fw.stop_experiment(args.name)
        print(f"Stopped experiment '{args.name}'")
    elif args.command == "assign":
        exp = fw.get_experiment(args.experiment)
        if not exp:
            print(f"Experiment not found: {args.experiment}", file=sys.stderr)
            sys.exit(1)
        variant = fw.assign_variant(exp, args.user_id)
        print(f"User '{args.user_id}' assigned to variant: {variant}")
    elif args.command == "convert":
        recorded = fw.record_conversion(args.experiment, args.user_id, value=args.value)
        print(f"Conversion {'recorded' if recorded else 'already exists'} for '{args.user_id}'")
    elif args.command == "stats":
        exp = fw.get_experiment(args.experiment)
        if not exp:
            print(f"Experiment not found", file=sys.stderr)
            sys.exit(1)
        sig = fw.calculate_significance(exp, confidence=args.confidence)
        print(json.dumps({"variant_stats": sig.variant_stats, "significance": {
            "chi2_stat": sig.chi2_stat, "p_value": sig.p_value,
            "is_significant": sig.is_significant, "reason": sig.reason,
            "recommended_winner": sig.recommended_winner,
        }}, indent=2))
    elif args.command == "winner":
        w = fw.get_winning_variant(args.experiment, confidence=args.confidence)
        print(f"Winner: {w}" if w else "No significant winner yet")
    elif args.command == "export":
        result = fw.export_results(args.experiment)
        print(json.dumps(result, indent=2))
    elif args.command == "list":
        exps = fw.list_experiments()
        for e in exps:
            print(f"  {e['name']:<30} {e['status']:<10} {e['metric']:<15} winner={e['winner'] or 'TBD'}")


if __name__ == "__main__":
    main()
