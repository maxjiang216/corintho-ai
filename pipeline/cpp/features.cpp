// Per-position features for the architecture experiments, computed by the
// engine's own rules and called from Python through ctypes
// (pipeline/arch/features.py; worklog/2026-09-25-nn-architectures).
//
//   corintho_features(states, rows, stride, legal, lines)
//
// states  float32 [rows * stride][70], network inputs as writeGameState
//         writes them; only every stride-th row is read (stride 8 skips the
//         seven symmetric copies of each position in a samples file)
// legal   uint8 [rows][12]: move j is legal when bit j % 8 of byte j / 8 is
// set lines   uint16 [rows][3]: spaces in a line, by top type
// (Game::lineSpaces)
//
// The input's reserves are canonized (the player to move first), so the
// position is rebuilt with player 0 to move; legality does not depend on
// which player that is.

#include <cstdint>
#include <cstring>

#include "game.h"
#include "util.h"

extern "C" void corintho_features(const float *states, int64_t rows,
                                  int64_t stride, uint8_t *legal,
                                  uint16_t *lines) {
#pragma omp parallel for schedule(static)
  for (int64_t r = 0; r < rows; ++r) {
    const float *s = states + r * stride * kGameStateSize;
    int32_t board[4 * kBoardSize];
    for (int32_t i = 0; i < 4 * kBoardSize; ++i)
      board[i] = s[i] != 0.0F ? 1 : 0;
    int32_t pieces[6];
    for (int32_t i = 0; i < 6; ++i)
      pieces[i] = static_cast<int32_t>(s[4 * kBoardSize + i] * 4.0F + 0.5F);
    const Game game{board, 0, pieces};
    std::bitset<kNumMoves> mask;
    game.getLegalMoves(mask);
    uint8_t *out = legal + r * 12;
    std::memset(out, 0, 12);
    for (int32_t j = 0; j < kNumMoves; ++j)
      if (mask[j])
        out[j / 8] |= static_cast<uint8_t>(1U << (j % 8));
    game.lineSpaces(lines + r * 3);
  }
}
