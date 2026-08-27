import pytest

from incident_agent.fixtures import FixtureError, load_metrics, search_logs


def test_fixture_metrics_and_log_search():
    metrics = load_metrics("demo-oom")
    assert metrics["gpu_memory_utilization"] == 0.97
    assert search_logs("demo-oom", "out of memory", limit=1)


def test_fixture_allowlist_and_limits():
    with pytest.raises(FixtureError):
        load_metrics("../../etc/passwd")
    with pytest.raises(FixtureError):
        search_logs("demo-oom", "")
    assert len(search_logs("demo-oom", "INFO", limit=1000)) <= 50

