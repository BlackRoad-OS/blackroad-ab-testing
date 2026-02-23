import pytest
from src.ab_testing import ABTestingFramework, Experiment, Variant, SignificanceResult


def make_fw(tmp_path):
    return ABTestingFramework(db_path=str(tmp_path / "test.db"))


def make_exp():
    return Experiment(
        name="test-exp",
        variants=[Variant("control", 50.0), Variant("treatment", 50.0)],
        description="Test experiment",
    )


def test_create_and_get_experiment(tmp_path):
    fw = make_fw(tmp_path)
    exp = make_exp()
    fw.create_experiment(exp)
    fetched = fw.get_experiment("test-exp")
    assert fetched is not None
    assert fetched.name == "test-exp"
    assert len(fetched.variants) == 2


def test_assign_variant_consistent(tmp_path):
    fw = make_fw(tmp_path)
    exp = make_exp()
    fw.create_experiment(exp)
    v1 = fw.assign_variant(exp, "user-001")
    v2 = fw.assign_variant(exp, "user-001")
    assert v1 == v2  # same user always gets same variant
    assert v1 in ["control", "treatment"]


def test_assign_variant_distribution(tmp_path):
    fw = make_fw(tmp_path)
    exp = make_exp()
    fw.create_experiment(exp)
    variants = [fw.assign_variant(exp, f"user-{i}") for i in range(1000)]
    control_pct = variants.count("control") / 1000 * 100
    assert 40 < control_pct < 60  # approximately 50/50


def test_record_conversion(tmp_path):
    fw = make_fw(tmp_path)
    exp = make_exp()
    fw.create_experiment(exp)
    fw.assign_variant(exp, "user-conv-1")
    result = fw.record_conversion("test-exp", "user-conv-1")
    assert result is True
    result2 = fw.record_conversion("test-exp", "user-conv-1")
    assert result2 is False  # duplicate


def test_calculate_significance(tmp_path):
    fw = make_fw(tmp_path)
    exp = make_exp()
    fw.create_experiment(exp)
    # Assign 200 users, convert heavily in treatment
    for i in range(200):
        v = fw.assign_variant(exp, f"u{i}")
    # Record conversions: control gets 10%, treatment gets 50%
    for i in range(200):
        v = fw.assign_variant(exp, f"u{i}")
        if v == "treatment" and i % 2 == 0:
            try:
                fw.record_conversion("test-exp", f"u{i}")
            except Exception:
                pass
        elif v == "control" and i % 10 == 0:
            try:
                fw.record_conversion("test-exp", f"u{i}")
            except Exception:
                pass
    sig = fw.calculate_significance(exp)
    assert isinstance(sig, SignificanceResult)
    assert hasattr(sig, "chi2_stat")


def test_export_results(tmp_path):
    fw = make_fw(tmp_path)
    exp = make_exp()
    fw.create_experiment(exp)
    fw.assign_variant(exp, "u1")
    result = fw.export_results("test-exp")
    assert "experiment" in result
    assert "variant_stats" in result
    assert "significance" in result


def test_list_experiments(tmp_path):
    fw = make_fw(tmp_path)
    fw.create_experiment(make_exp())
    fw.create_experiment(Experiment("exp2", [Variant("a", 50.0), Variant("b", 50.0)]))
    exps = fw.list_experiments()
    assert len(exps) == 2


def test_duplicate_experiment_raises(tmp_path):
    fw = make_fw(tmp_path)
    fw.create_experiment(make_exp())
    with pytest.raises(ValueError, match="already exists"):
        fw.create_experiment(make_exp())


def test_invalid_traffic_sum_raises():
    with pytest.raises(ValueError, match="sum to 100"):
        Experiment("bad", [Variant("a", 30.0), Variant("b", 30.0)])


def test_start_stop_experiment(tmp_path):
    fw = make_fw(tmp_path)
    exp = make_exp()
    fw.create_experiment(exp)
    started = fw.start_experiment("test-exp")
    assert started.status == "running"
    stopped = fw.stop_experiment("test-exp")
    assert stopped.status == "stopped"


def test_no_winner_insufficient_data(tmp_path):
    fw = make_fw(tmp_path)
    exp = make_exp()
    fw.create_experiment(exp)
    fw.assign_variant(exp, "u1")
    winner = fw.get_winning_variant("test-exp")
    assert winner is None
