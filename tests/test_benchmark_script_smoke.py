"""Fast benchmark-harness smoke check; does not run the 300k workload in CI."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def test_synthetic_external_benchmark_harness_smoke(tmp_path):
    path = Path(__file__).resolve().parents[1] / "benchmarks" / "linked_300k.py"
    spec = spec_from_file_location("geco_benchmark_300k", path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    project = module.synthetic_project(tmp_path / "bench.geco", 30)
    result = module.exercise(
        project, page_size=5, repeats=1,
        filter_field="category", filter_value="A",
    )
    assert result["units"] == 30
    assert result["topic_page"]["median_seconds"] >= 0
    assert result["view_page"]["median_seconds"] >= 0
