// Fixed-depth benchmark of ab_match's AlphaBeta: nodes, nodes/s and a
// digest of the moves chosen (worklog 2026-09-25-nn-architectures, entry 24)
//
//   ab_bench NET POSITIONS.bin MIN_P MAX_P COUNT
//   e.g. AB_MAX_DEPTH=6 build/ab_bench NET ../bench/results/solver-positions.bin 32 40 300
//
// Without AB_MAX_DEPTH each search runs until a forced result. Node counts
// and the digest are deterministic at a fixed depth, so engine changes that
// should not change the search can be checked exactly, and timed. Speed and
// node counts do not need a trained network: arch/random_small_net.py
// writes a random one.
#define main ab_match_main
#include "ab_match.cpp"
#undef main

int main(int argc, char **argv) {
  if (argc < 6) {
    std::fprintf(stderr, "usage: ab_bench NET POSITIONS.bin MIN_P MAX_P COUNT\n");
    return 1;
  }
  const SmallNet net{argv[1]};
  FILE *f = std::fopen(argv[2], "rb");
  const int min_p = std::atoi(argv[3]), max_p = std::atoi(argv[4]);
  const int count = std::atoi(argv[5]);
  std::vector<Game> games;
  uint8_t s[71];
  while (std::fread(s, 1, 71, f) == 71 && static_cast<int>(games.size()) < count) {
    int32_t board[64], pieces[6], occ = 0, res = 0;
    for (int i = 0; i < 64; ++i) board[i] = s[i] != 0;
    for (int sp = 0; sp < 16; ++sp) occ += (s[4 * sp] | s[4 * sp + 1] | s[4 * sp + 2]) != 0;
    for (int i = 0; i < 6; ++i) res += pieces[i] = s[64 + i];
    const int p = 2 * res + occ;
    if (p < min_p || p > max_p) continue;
    Game g(board, 0, pieces);
    MoveMask legal;
    g.getLegalMoves(legal);
    if (legal.count() < 2) continue;
    games.push_back(g);
  }
  AlphaBeta ab{net, 20};
  uint64_t nodes = 0, digest = 1469598103934665603ULL;
  long depth_sum = 0;
  const auto t0 = Clock::now();
  for (const Game &g : games) {
    const AlphaBeta::Result r = ab.choose(g, 1e9);
    nodes += r.nodes;
    depth_sum += r.depth;
    digest = (digest ^ static_cast<uint64_t>(r.move + 1)) * 1099511628211ULL;
  }
  const double t = since(t0);
  std::printf("positions %zu, depth %.1f, nodes %llu, %.2f s, %.0f nodes/s, moves digest %016llx\n",
              games.size(), double(depth_sum) / games.size(), (unsigned long long)nodes, t,
              nodes / t, (unsigned long long)digest);
}
