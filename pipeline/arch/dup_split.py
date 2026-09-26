"""Where repeated network evaluations come from (worklog
2026-09-25-nn-architectures, entry 17).

    CORINTHO_DUP_DUMP=dup.bin build/corintho_play train ...
    .venv/bin/python arch/dup_split.py dup.bin

Each row of the dump is one position sent to the network. A search is
identified by its game and its root's horizon P (P falls every move), and its
player by the parity of the search's rank within the game. A row is either new
to its game, or repeats a position evaluated earlier in the same search (a
transposition), in an earlier search of the same player, or only in the other
player's searches.
"""

import sys

import numpy as np

RECORD = np.dtype(
    [
        ("game", "<u4"),
        ("call", "<u4"),
        ("hash", "<u8"),
        ("canonical", "<u8"),
        ("root_p", "u1"),
        ("leaf_p", "u1"),
        ("pad", "V6"),
    ]
)


def search_rank(game, root_p):
    """Rank of each row's search within its game, 0 for the first move"""
    key = game.astype(np.int64) * 256 + root_p
    searches, inverse = np.unique(key, return_inverse=True)
    rank = np.empty(len(searches), np.int64)
    games = searches // 256
    # Within a game, searches in falling root P
    order = np.lexsort((-(searches % 256), games))
    starts = np.r_[0, np.flatnonzero(np.diff(games[order])) + 1]
    counts = np.diff(np.r_[starts, len(order)])
    rank[order] = np.arange(len(order)) - np.repeat(starts, counts)
    return rank[inverse]


def first_seen(index, *keys):
    """Whether each row is the first, in file order, with its keys"""
    order = np.lexsort((index,) + tuple(reversed(keys)))
    repeat = np.ones(len(index), bool)
    for k in keys:
        s = k[order]
        repeat[1:] &= s[1:] == s[:-1]
    repeat[0] = False
    first = np.empty(len(index), bool)
    first[order] = ~repeat
    return first


def main(path):
    d = np.fromfile(path, dtype=RECORD)
    d = d[d["root_p"] != 255]
    index = np.arange(len(d))
    search = search_rank(d["game"], d["root_p"])
    player = search & 1
    game = d["game"]
    for label, h in (("exact", d["hash"]), ("up to symmetry", d["canonical"])):
        in_game = first_seen(index, game, h)
        in_player = first_seen(index, game, h, player)
        in_search = first_seen(index, game, h, search)
        print(f"{label}: {len(d):,} rows")
        print(f"  new to the game:                   {in_game.mean():.1%}")
        print(
            f"  repeat within the same search:     {(~in_search).mean():.1%}"
        )
        print(
            "  repeat of the same player's earlier search: "
            f"{(in_search & ~in_player).mean():.1%}"
        )
        print(
            "  seen before only by the other player:       "
            f"{(in_player & ~in_game).mean():.1%}"
        )


if __name__ == "__main__":
    main(sys.argv[1])
