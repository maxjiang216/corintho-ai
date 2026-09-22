"""Promotion gate for a new generation.

A new model replaces the best model only if it is *statistically* better, not
merely ahead on the sample. The test is on decisive games only: draws carry no
information about which side is stronger, so they are discarded rather than
counted as half a win. What remains is a binomial question -- of the games that
were decided, did the new agent win more than half? -- and the gate is the
lower bound of a Wilson score interval on that proportion.

Why this rather than a fixed score threshold:

  The old gate was `score > 0.52`, with score counting draws as 0.5. At the
  1,600 test games used for the recorded run that is roughly a one-sided 94%
  test, so it was not far off. But it is fixed, and significance is not: the
  same 0.52 is a much stronger demand at 6,400 games and a much weaker one at
  400. Since the plan is to vary the number of test games, a threshold that
  does not move with the sample size will silently change how strict the gate
  is. A confidence level does not have that problem.

Wilson is used in preference to the normal approximation because it behaves
correctly for small samples and for proportions near 0 or 1. At n in the
thousands with p near 0.5 the two agree to about three decimals; the
difference only matters if the test size is ever reduced.
"""

import math

# Two-sided z for a few common confidence levels. Avoids a scipy dependency;
# the training image already carries enough.
_Z = {0.80: 1.2815515655446004,
      0.90: 1.6448536269514722,
      0.95: 1.9599639845400545,
      0.98: 2.3263478740408408,
      0.99: 2.5758293035489004}


def z_for(confidence):
    """Two-sided z value for a confidence level."""
    if confidence in _Z:
        return _Z[confidence]
    raise ValueError(
        f"confidence {confidence} not tabulated; use one of {sorted(_Z)}"
    )


def wilson_bounds(wins, decisive, confidence=0.95):
    """Wilson score interval for wins/decisive.

    Returns (low, high). With no decisive games the interval is (0.0, 1.0):
    nothing was learned, so nothing is excluded.
    """
    if decisive <= 0:
        return 0.0, 1.0
    z = z_for(confidence)
    n = float(decisive)
    p = wins / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    # Clamp: rounding can push the bound a hair outside [0, 1] at w = 0 or
    # w = n (observed -2.8e-17 for 0 of 10), and a probability bound outside
    # [0, 1] is meaningless.
    return max(0.0, center - margin), min(1.0, center + margin)


def should_promote(wins, draws, games, confidence=0.95):
    """Decide whether the new agent replaces the best agent.

    Promote only when the *whole* confidence interval on the decisive-game win
    rate sits above 0.5 -- that is, when the lower bound exceeds 0.5. Being
    ahead on the sample is not enough.

    Returns (promote, info) where info is a dict suitable for logging.
    """
    losses = games - wins - draws
    decisive = wins + losses
    low, high = wilson_bounds(wins, decisive, confidence)
    promote = low > 0.5
    return promote, {
        "games": games,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "decisive": decisive,
        "win_rate_decisive": (wins / decisive) if decisive else float("nan"),
        "score_with_draws": ((wins + 0.5 * draws) / games) if games else float("nan"),
        "ci_low": low,
        "ci_high": high,
        "confidence": confidence,
        "promote": promote,
    }


def describe(info):
    """One-line summary for score.txt and the logs."""
    return (
        "new agent: {wins}W {draws}D {losses}L of {games}  "
        "decisive win rate {win_rate_decisive:.4f}  "
        "{confidence:.0%} CI [{ci_low:.4f}, {ci_high:.4f}]  "
        "score(with draws) {score_with_draws:.4f}  "
        "-> {verdict}".format(verdict="PROMOTE" if info["promote"] else "reject",
                              **info)
    )
