// Cost and consistency of Solver::playOut (worklog
// 2026-09-25-nn-architectures, entry 19).
//
//   solver_playout POSITIONS.bin P CAP THREADS [LOG2_TABLE]
//
// POSITIONS.bin: rows of 71 bytes as for solver_bench. Takes the positions
// with horizon exactly P and, on THREADS threads each with one persistent
// solver (as SolverPool), solves each position and then plays it out.
// Prints the time per position for the solve and the playout, line lengths,
// and checks every line: values alternate in sign, the first equals the
// solve, and the game ends with the result the values promise.

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "game.h"
#include "solver.h"
#include "util.h"

int main(int argc, char **argv) {
  if (argc < 5) {
    std::fprintf(stderr, "usage: solver_playout POSITIONS.bin P CAP THREADS "
                         "[LOG2_TABLE]\n");
    return 1;
  }
  FILE *f = std::fopen(argv[1], "rb");
  if (f == nullptr)
    return 1;
  const int32_t want_p = std::atoi(argv[2]);
  const uint64_t cap = std::strtoull(argv[3], nullptr, 10);
  omp_set_num_threads(std::atoi(argv[4]));
  const int32_t log2_table = argc > 5 ? std::atoi(argv[5]) : 22;
  std::vector<Game> games;
  uint8_t s[71];
  while (std::fread(s, 1, 71, f) == 71) {
    int32_t board[4 * kBoardSize];
    int32_t occupied = 0, reserves = 0, pieces[6];
    for (int32_t i = 0; i < 4 * kBoardSize; ++i)
      board[i] = s[i] != 0;
    for (int32_t sp = 0; sp < kBoardSize; ++sp)
      occupied += (s[4 * sp] | s[4 * sp + 1] | s[4 * sp + 2]) != 0;
    for (int32_t i = 0; i < 6; ++i)
      reserves += pieces[i] = s[4 * kBoardSize + i];
    if (2 * reserves + occupied == want_p)
      games.emplace_back(board, 0, pieces);
  }
  std::fclose(f);
  struct Out {
    double solve_s, play_s;
    int32_t value, length;
    bool capped, consistent;
  };
  std::vector<Out> out(games.size());
  const auto t_all = std::chrono::steady_clock::now();
#pragma omp parallel
  {
    Solver solver{log2_table};
    std::vector<Solver::LineStep> line;
#pragma omp for schedule(dynamic, 1)
    for (size_t i = 0; i < games.size(); ++i) {
      const auto t0 = std::chrono::steady_clock::now();
      const int32_t v = solver.solve(games[i], cap);
      const auto t1 = std::chrono::steady_clock::now();
      const bool ok =
          v != Solver::kUnknown && solver.playOut(games[i], cap, line);
      const auto t2 = std::chrono::steady_clock::now();
      Out &o = out[i];
      o.solve_s = std::chrono::duration<double>(t1 - t0).count();
      o.play_s = std::chrono::duration<double>(t2 - t1).count();
      o.value = v;
      o.length = static_cast<int32_t>(line.size());
      o.capped = !ok;
      o.consistent = false;
      if (ok && !line.empty() && line[0].value == v) {
        bool alternate = true;
        for (size_t k = 1; k < line.size(); ++k)
          alternate &= line[k].value == -line[k - 1].value;
        // After the last move: no legal moves; a line stands exactly when
        // the last mover won
        Game end = line.back().game;
        end.doMove(line.back().move);
        MoveMask legal;
        const bool lines = end.getLegalMoves(legal);
        const bool ends = legal.count() == 0;
        const int32_t last = line.back().value;
        o.consistent = alternate && ends &&
                       ((last == 1 && lines) || (last == 0 && !lines));
      }
    }
  }
  const double wall =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t_all)
          .count();
  std::vector<double> solve_s, play_s;
  int32_t capped = 0, inconsistent = 0, counts[3] = {0, 0, 0};
  int64_t length_sum[3] = {0, 0, 0};
  for (const Out &o : out) {
    if (o.capped) {
      ++capped;
      continue;
    }
    inconsistent += !o.consistent;
    solve_s.push_back(o.solve_s);
    play_s.push_back(o.play_s);
    ++counts[o.value + 1];
    length_sum[o.value + 1] += o.length;
  }
  std::sort(solve_s.begin(), solve_s.end());
  std::sort(play_s.begin(), play_s.end());
  auto q = [](const std::vector<double> &v, double p) {
    return v.empty() ? 0.0 : v[static_cast<size_t>(p * (v.size() - 1))];
  };
  double solve_sum = 0, play_sum = 0;
  for (double x : solve_s)
    solve_sum += x;
  for (double x : play_s)
    play_sum += x;
  std::printf("positions %zu at P %d; capped %d; inconsistent lines %d\n",
              games.size(), want_p, capped, inconsistent);
  std::printf("solve ms: mean %.2f p50 %.2f p90 %.2f max %.2f\n",
              1e3 * solve_sum / std::max<size_t>(1, solve_s.size()),
              1e3 * q(solve_s, 0.5), 1e3 * q(solve_s, 0.9),
              1e3 * q(solve_s, 1.0));
  std::printf("playout ms: mean %.2f p50 %.2f p90 %.2f max %.2f\n",
              1e3 * play_sum / std::max<size_t>(1, play_s.size()),
              1e3 * q(play_s, 0.5), 1e3 * q(play_s, 0.9),
              1e3 * q(play_s, 1.0));
  const char *names[3] = {"lost", "drawn", "won"};
  for (int32_t k = 0; k < 3; ++k)
    std::printf(
        "%s: %d positions, mean line %.2f plies\n", names[k], counts[k],
        counts[k] ? static_cast<double>(length_sum[k]) / counts[k] : 0.0);
  std::printf("wall %.2f s\n", wall);
  return 0;
}
