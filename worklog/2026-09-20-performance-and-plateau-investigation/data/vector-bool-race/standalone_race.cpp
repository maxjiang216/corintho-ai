// Can the vector<bool> race fire at all? Vary scheduling and contention.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <omp.h>

enum Sched { kStatic, kStatic1, kDynamic };

template <typename Vec>
long run(int games, int threads, int trials, Sched s) {
  long total_lost = 0;
  for (int t = 0; t < trials; ++t) {
    Vec is_done(games, false);
    omp_set_num_threads(threads);
    if (s == kStatic) {
#pragma omp parallel for schedule(static)
      for (int i = 0; i < games; ++i) is_done[i] = true;
    } else if (s == kStatic1) {
#pragma omp parallel for schedule(static, 1)
      for (int i = 0; i < games; ++i) is_done[i] = true;
    } else {
#pragma omp parallel for schedule(dynamic, 1)
      for (int i = 0; i < games; ++i) is_done[i] = true;
    }
    for (int i = 0; i < games; ++i)
      if (!is_done[i]) ++total_lost;
  }
  return total_lost;
}

int main(int argc, char **argv) {
  int games = argc > 1 ? atoi(argv[1]) : 3000;
  int threads = argc > 2 ? atoi(argv[2]) : 14;
  int trials = argc > 3 ? atoi(argv[3]) : 500;
  const char *names[] = {"static (default)", "static,1 (interleaved)",
                         "dynamic,1"};
  for (int s = 0; s < 3; ++s) {
    long b = run<std::vector<bool>>(games, threads, trials, (Sched)s);
    long u = run<std::vector<uint8_t>>(games, threads, trials, (Sched)s);
    printf("%-24s  vector<bool> lost=%-8ld (%.5f%%)   vector<uint8_t> lost=%ld\n",
           names[s], b, 100.0 * b / ((double)games * trials), u);
  }
  return 0;
}
