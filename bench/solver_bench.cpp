// Endgame solver benchmark and correctness gate (worklog
// 2026-09-25-nn-architectures, entry 08).
//
//   solver_bench POSITIONS.bin MAX_P CAP THREADS [RESULTS.tsv] [LOG2_TABLE]
//
// POSITIONS.bin: rows of 71 bytes, the 70 network inputs x 4 (uint8) and a
// value target (bench/results/solver-positions.bin: 9171 self-play positions,
// up to 300 per horizon P). Solves every position with P <= MAX_P from an
// empty table, capped at CAP positions visited, and prints per P: count,
// solved, median / p90 / max microseconds and median nodes over the solved,
// plus totals. RESULTS.tsv gets one line per position (index, P, result,
// nodes, us) for the gate: an optimization may solve more positions, but
// every position solved before must keep its result (bench/solver_gate.py).

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "game.h"
#include "solver.h"
#include "util.h"

int main(int argc, char **argv) {
  if (argc < 5) {
    std::fprintf(stderr,
                 "usage: solver_bench POSITIONS.bin MAX_P CAP THREADS "
                 "[RESULTS.tsv]\n");
    return 1;
  }
  FILE *f = std::fopen(argv[1], "rb");
  if (f == nullptr)
    return 1;
  std::vector<uint8_t> raw;
  uint8_t buf[71];
  while (std::fread(buf, 1, 71, f) == 71)
    raw.insert(raw.end(), buf, buf + 71);
  std::fclose(f);
  const int32_t max_p = std::atoi(argv[2]);
  const uint64_t cap = std::strtoull(argv[3], nullptr, 10);
  omp_set_num_threads(std::atoi(argv[4]));

  struct Row {
    int32_t index, p, result;
    uint64_t nodes, us;
  };
  std::vector<Row> rows;
  std::vector<Game> games;
  for (size_t r = 0; r * 71 < raw.size(); ++r) {
    const uint8_t *s = raw.data() + r * 71;
    int32_t board[4 * kBoardSize];
    int32_t occupied = 0, reserves = 0;
    for (int32_t i = 0; i < 4 * kBoardSize; ++i)
      board[i] = s[i] != 0;
    for (int32_t sp = 0; sp < kBoardSize; ++sp)
      occupied += (s[4 * sp] | s[4 * sp + 1] | s[4 * sp + 2]) != 0;
    int32_t pieces[6];
    for (int32_t i = 0; i < 6; ++i)
      reserves += pieces[i] = s[4 * kBoardSize + i];
    const int32_t p = 2 * reserves + occupied;
    if (p > max_p)
      continue;
    rows.push_back({static_cast<int32_t>(r), p, 0, 0, 0});
    games.emplace_back(board, 0, pieces);
  }
  const auto t_all = std::chrono::steady_clock::now();
#pragma omp parallel
  {
    Solver solver{argc > 6 ? std::atoi(argv[6]) : 20};
#pragma omp for schedule(dynamic, 1)
    for (size_t i = 0; i < rows.size(); ++i) {
      solver.clear();
      const auto t0 = std::chrono::steady_clock::now();
      rows[i].result = solver.solve(games[i], cap);
      rows[i].us = static_cast<uint64_t>(
          std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::steady_clock::now() - t0)
              .count());
      rows[i].nodes = solver.nodes();
    }
  }
  const double wall = std::chrono::duration<double>(
                          std::chrono::steady_clock::now() - t_all)
                          .count();

  std::printf(" P |   n | solved | median us |   p90 us |    max us | median nodes\n");
  uint64_t total_nodes = 0, total_us = 0, solved_all = 0;
  for (int32_t p = 0; p <= max_p; ++p) {
    std::vector<uint64_t> us, nodes;
    int32_t n = 0;
    for (const Row &r : rows) {
      if (r.p != p)
        continue;
      ++n;
      total_nodes += r.nodes;
      total_us += r.us;
      if (r.result != Solver::kUnknown) {
        us.push_back(r.us);
        nodes.push_back(r.nodes);
      }
    }
    if (n == 0)
      continue;
    solved_all += us.size();
    std::sort(us.begin(), us.end());
    std::sort(nodes.begin(), nodes.end());
    auto q = [](const std::vector<uint64_t> &v, double f) {
      return v.empty() ? uint64_t{0} : v[static_cast<size_t>(f * (v.size() - 1))];
    };
    std::printf("%2d | %3d | %5.1f%% | %9lu | %8lu | %9lu | %12lu\n", p, n,
                100.0 * us.size() / n, q(us, 0.5), q(us, 0.9),
                us.empty() ? 0UL : us.back(), q(nodes, 0.5));
  }
  std::printf("positions %zu, solved %lu, nodes %lu, solve time %.2f s "
              "(summed), wall %.2f s\n",
              rows.size(), solved_all, total_nodes, total_us / 1e6, wall);
  if (argc > 5 && std::string(argv[5]) != "-") {
    FILE *o = std::fopen(argv[5], "w");
    std::fprintf(o, "index\tP\tresult\tnodes\tus\n");
    for (const Row &r : rows)
      std::fprintf(o, "%d\t%d\t%d\t%lu\t%lu\n", r.index, r.p, r.result, r.nodes,
                   r.us);
    std::fclose(o);
  }
  return 0;
}
