#include <iostream>
// Differential test: the implementation under test against the frozen reference.
//
// digest_game tells you THAT behaviour changed. This tells you WHERE, in terms
// you can check against the written rules -- which is the second half of the
// oracle (see worklog/RULES-CHECKLIST.md). The reference is descriptive, not
// normative: when the two disagree, this output is what you read the rulebook
// against to decide which side is wrong.
//
// The corpus is built from real play, not arbitrary bit patterns. Some of the
// optimizations rest on invariants that hold only for reachable positions --
// "an empty space is never frozen", for instance -- so unreachable states would
// produce false failures.
//
// Coverage is reported rather than assumed: a corpus that never contains a
// capital line has not tested the capital-line special case, however large it is.

#include <cstdint>

#include <bitset>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "bench_util.h"
#include "game.h"
#include "move.h"
#include "util.h"

namespace {

struct Coverage {
  int32_t positions{0};
  int32_t with_lines{0};
  int32_t terminal{0};
  int32_t has_frozen{0};
  int32_t has_capital{0};
  int32_t capital_on_top{0};
  int32_t piece_exhausted{0};
  int32_t few_legal{0};   // fewer than 8 legal moves
  int32_t many_legal{0};  // more than 40
};

/// @brief Decode a move ID into something a human can check against the rules.
void describeMove(int32_t id, char *out, size_t n) {
  Move move{id};
  if (move.move_type() == Move::MoveType::kPlace) {
    const char *piece = move.piece_type() == kBase      ? "base"
                        : move.piece_type() == kColumn  ? "column"
                                                        : "capital";
    std::snprintf(out, n, "place %-7s at %c%d", piece,
                  getColName(move.col_to()), 4 - move.row_to());
  } else {
    std::snprintf(out, n, "move %c%d -> %c%d", getColName(move.col_from()),
                  4 - move.row_from(), getColName(move.col_to()),
                  4 - move.row_to());
  }
}

/// @brief Random legal play from the starting position, recording every state.
/// @details Reachable positions only, deliberately -- see the file comment.
void collect(std::vector<Game> &corpus, size_t target, uint64_t seed) {
  uint64_t rng = seed;
  std::bitset<kNumMoves> legal;
  while (corpus.size() < target) {
    Game game;
    while (corpus.size() < target) {
      corpus.push_back(game);
      game.getLegalMoves(legal);
      const size_t count = legal.count();
      if (count == 0)
        break;
      size_t pick = bench::splitmix64(rng) % count;
      int32_t move_id = 0;
      for (int32_t i = 0; i < kNumMoves; ++i) {
        if (legal[i]) {
          if (pick == 0) {
            move_id = i;
            break;
          }
          --pick;
        }
      }
      game.doMove(move_id);
    }
  }
}

void measureCoverage(const Game &game, const std::bitset<kNumMoves> &legal,
                     bool lines, Coverage &cov) {
  ++cov.positions;
  if (lines)
    ++cov.with_lines;
  const int32_t count = static_cast<int32_t>(legal.count());
  if (count == 0)
    ++cov.terminal;
  if (count > 0 && count < 8)
    ++cov.few_legal;
  if (count > 40)
    ++cov.many_legal;

  // Read the board through the public game-state encoding, so this stays
  // independent of Game's internals as they change across the stages.
  float state[kGameStateSize];
  game.writeGameState(state);
  bool frozen = false;
  bool capital = false;
  bool cap_top = false;
  for (int32_t sp = 0; sp < kBoardSize; ++sp) {
    const bool b = state[sp * 4 + kBase] != 0.0F;
    const bool c = state[sp * 4 + kColumn] != 0.0F;
    const bool a = state[sp * 4 + kCapital] != 0.0F;
    if (state[sp * 4 + kFrozen] != 0.0F)
      frozen = true;
    if (a) {
      capital = true;
      cap_top = true;
    }
    (void)b;
    (void)c;
  }
  if (frozen)
    ++cov.has_frozen;
  if (capital)
    ++cov.has_capital;
  if (cap_top)
    ++cov.capital_on_top;
  for (int32_t i = 0; i < 6; ++i) {
    if (state[4 * kBoardSize + i] == 0.0F) {
      ++cov.piece_exhausted;
      break;
    }
  }
}

}  // namespace

int main(int argc, char **argv) {
  const size_t corpus_size = argc > 1 ? std::stoul(argv[1]) : 200000;
  const uint64_t seed = argc > 2 ? std::stoull(argv[2]) : 0x1234ABCDULL;

  std::printf("Differential verification: getLegalMoves vs "
              "getLegalMovesReference\n");
  std::printf("corpus %zu reachable positions, seed 0x%llx\n\n", corpus_size,
              static_cast<unsigned long long>(seed));

  std::vector<Game> corpus;
  corpus.reserve(corpus_size);
  collect(corpus, corpus_size, seed);

  Coverage cov;
  int32_t mismatches = 0;
  std::bitset<kNumMoves> actual;
  std::bitset<kNumMoves> expected;

  for (size_t idx = 0; idx < corpus.size(); ++idx) {
    const Game &game = corpus[idx];
    const bool lines_actual = game.getLegalMoves(actual);
    const bool lines_expected = game.getLegalMovesReference(expected);
    measureCoverage(game, expected, lines_expected, cov);

    if (actual == expected && lines_actual == lines_expected)
      continue;

    ++mismatches;
    if (mismatches > 3) {
      continue;  // Three worked examples is enough to debug from.
    }
    std::printf("=== MISMATCH at corpus position %zu ===\n\n", idx);
    std::printf("%s\n\n", "board:");
    std::cout << game << "\n\n";
    if (lines_actual != lines_expected) {
      std::printf("  is_lines: reference=%s  actual=%s\n\n",
                  lines_expected ? "true" : "false",
                  lines_actual ? "true" : "false");
    }
    const std::bitset<kNumMoves> diff = actual ^ expected;
    std::printf("  %zu move(s) disagree:\n", diff.count());
    char desc[64];
    for (int32_t i = 0; i < kNumMoves; ++i) {
      if (!diff[i])
        continue;
      describeMove(i, desc, sizeof(desc));
      std::printf("    id %2d  %-26s  reference=%-8s actual=%s\n", i, desc,
                  expected[i] ? "LEGAL" : "illegal",
                  actual[i] ? "LEGAL" : "illegal");
    }
    std::printf("\n  Check against worklog/RULES-CHECKLIST.md and the rules\n"
                "  overlay in web/corintho.js to decide which side is right.\n\n");
  }

  std::printf("Corpus coverage\n");
  std::printf("  positions              %8d\n", cov.positions);
  std::printf("  with lines             %8d (%5.2f%%)\n", cov.with_lines,
              100.0 * cov.with_lines / cov.positions);
  std::printf("  terminal               %8d (%5.2f%%)\n", cov.terminal,
              100.0 * cov.terminal / cov.positions);
  std::printf("  containing frozen      %8d (%5.2f%%)\n", cov.has_frozen,
              100.0 * cov.has_frozen / cov.positions);
  std::printf("  containing a capital   %8d (%5.2f%%)\n", cov.has_capital,
              100.0 * cov.has_capital / cov.positions);
  std::printf("  a piece exhausted      %8d (%5.2f%%)\n", cov.piece_exhausted,
              100.0 * cov.piece_exhausted / cov.positions);
  std::printf("  under 8 legal moves    %8d (%5.2f%%)\n", cov.few_legal,
              100.0 * cov.few_legal / cov.positions);
  std::printf("  over 40 legal moves    %8d (%5.2f%%)\n", cov.many_legal,
              100.0 * cov.many_legal / cov.positions);
  std::printf("\n");

  if (mismatches == 0) {
    std::printf("PASS: %zu positions, identical on every one.\n", corpus.size());
    return 0;
  }
  std::printf("FAIL: %d of %zu positions disagree.\n", mismatches,
              corpus.size());
  return 1;
}
