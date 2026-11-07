"""Phase 2 regression coverage for count-only pagination."""

import pytest

from geometric_coder.explore import deterministic_page, deterministic_page_count


@pytest.mark.parametrize(
    ("total", "size"),
    [(0, 1), (0, 5000), (1, 5000), (4999, 5000),
     (5000, 5000), (5001, 5000), (10001, 7)],
)
def test_page_count_matches_actual_pagination(total, size):
    expected = deterministic_page(total, page_size=size)
    assert deterministic_page_count(total, page_size=size) == expected.page_count


@pytest.mark.parametrize(
    ("total", "size"),
    [(-1, 5000), (1, 0), (1, -2)],
)
def test_page_count_rejects_invalid_arguments(total, size):
    with pytest.raises(ValueError):
        deterministic_page_count(total, page_size=size)


def test_count_only_does_not_allocate_random_permutation(monkeypatch):
    def reject_rng(*args, **kwargs):
        raise AssertionError("count-only pagination must not create an RNG")

    monkeypatch.setattr("geometric_coder.explore.np.random.default_rng", reject_rng)
    assert deterministic_page_count(10_000_000, page_size=5000) == 2000


def test_page_content_and_clamping_unchanged():
    first = deterministic_page(37, page_size=10, page_index=0, seed=23)
    last = deterministic_page(37, page_size=10, page_index=100, seed=23)
    assert first.page_index == 0
    assert last.page_index == 3
    assert first.page_count == last.page_count == deterministic_page_count(37, page_size=10)
    assert len(first.positions) == 10
    assert len(last.positions) == 7
    assert set(first.positions).isdisjoint(last.positions)
