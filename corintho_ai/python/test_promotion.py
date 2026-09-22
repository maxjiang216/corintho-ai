"""Tests for the promotion gate. Pure Python -- no cython, no TensorFlow."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from promotion import wilson_bounds, should_promote, z_for

def test_bounds_stay_in_unit_interval():
    for w, n in ((0, 10), (10, 10), (0, 1), (1, 1), (0, 0), (5, 10)):
        lo, hi = wilson_bounds(w, n)
        assert 0.0 <= lo <= hi <= 1.0, (w, n, lo, hi)

def test_no_decisive_games_learns_nothing():
    assert wilson_bounds(0, 0) == (0.0, 1.0)
    assert should_promote(0, 50, 50)[0] is False

def test_even_split_never_promotes():
    for n in (10, 100, 1600, 100000):
        assert should_promote(n // 2, 0, n)[0] is False, n

def test_interval_centred_and_narrows_with_n():
    prev = 1.0
    for n in (100, 400, 1600, 6400):
        lo, hi = wilson_bounds(n // 2, n)
        assert abs((lo + hi) / 2 - 0.5) < 1e-9
        half = (hi - lo) / 2
        assert half < prev, (n, half, prev)
        prev = half

def test_lower_bound_monotone_in_wins():
    prev = -1.0
    for w in range(700, 900):
        lo, _ = wilson_bounds(w, 1584)
        assert lo > prev
        prev = lo

def test_stricter_confidence_is_stricter():
    # the same record must not promote at a higher confidence if it failed lower
    for w in range(800, 900):
        a = should_promote(w, 16, 1600, 0.90)[0]
        b = should_promote(w, 16, 1600, 0.95)[0]
        c = should_promote(w, 16, 1600, 0.99)[0]
        assert a >= b >= c, (w, a, b, c)

def test_draws_are_discarded_not_counted():
    # Same decisive record, very different draw counts -> same verdict.
    base = should_promote(900, 0, 1600)
    many = should_promote(900, 600, 2200)
    assert base[0] == many[0]
    assert base[1]["ci_low"] == many[1]["ci_low"]

def test_scales_with_sample_size():
    # A 53% decisive rate is noise at 200 games and real at 20000.
    assert should_promote(106, 0, 200)[0] is False
    assert should_promote(10600, 0, 20000)[0] is True

def test_z_table():
    assert abs(z_for(0.95) - 1.959963984540054) < 1e-12
    for bad in (0.93, 0.5, 1.0):
        try:
            z_for(bad); assert False, bad
        except ValueError:
            pass

if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f(); print(f"  PASS  {f.__name__}")
    print(f"\n{len(fns)} tests passed")
