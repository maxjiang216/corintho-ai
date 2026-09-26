// Selection over ~17 children: linked list of scattered 64-byte nodes (today)
// vs the same stats in contiguous per-parent arrays (scalar, auto-vectorizable).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>
struct alignas(64) N { uint64_t game; N *parent, *next, *first; void *edges; float eval, denom; int16_t visits; int8_t result, id, depth, nlegal; bool allv; char pad[64-51]; };
struct Parent { N *first; float prob[48]; };
struct Soa { float w[48]; float n[48]; float prob[48]; uint8_t skip[48]; int k; };
int main() {
  const int P = PARENTS, K = 17;           // parents, children each
  std::mt19937 g(1);
  std::vector<N> pool(P * K * 2);        // big pool, children scattered through it
  std::vector<uint32_t> idx(pool.size()); for (uint32_t i = 0; i < idx.size(); ++i) idx[i] = i;
  std::shuffle(idx.begin(), idx.end(), g);
  std::vector<Parent> par(P); std::vector<Soa> soa(P);
  size_t next = 0;
  for (int p = 0; p < P; ++p) {
    N *prev = nullptr;
    soa[p].k = K;
    for (int c = 0; c < K; ++c) {
      N *n = &pool[idx[next++]];
      n->eval = (g() % 2000) / 1000.0f - 1; n->visits = 1 + g() % 200; n->result = 0; n->allv = false; n->next = nullptr;
      if (prev) prev->next = n; else par[p].first = n; prev = n;
      par[p].prob[c] = soa[p].prob[c] = (1 + g() % 511) / 8000.0f;
      soa[p].w[c] = n->eval; soa[p].n[c] = n->visits; soa[p].skip[c] = 0;
    }
  }
  std::vector<int> order(P * 20); for (auto &o : order) o = g() % P;   // random parent visits
  const float vs = 3.0f * std::sqrt(800.0f);
  volatile int sink = 0;
  auto run = [&](const char *name, auto fn) {
    double best = 1e9;
    for (int r = 0; r < 5; ++r) {
      auto t0 = std::chrono::steady_clock::now();
      for (int p : order) sink = sink + fn(p);
      best = std::min(best, std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count() / order.size());
    }
    printf("%-28s %6.1f ns / call  (%.2f ns / child)\n", name, best, best / K);
  };
  run("linked list (today)", [&](int p) {
    float mx = -1e30f; int bi = -1, i = 0;
    for (N *c = par[p].first; c; c = c->next, ++i) {
      if (c->result != 0 || c->allv) continue;
      float n = c->visits, u = -c->eval / n + par[p].prob[i] * vs / (n + 1);
      if (u > mx) { mx = u; bi = i; }
    }
    return bi; });
  run("parent arrays, scalar", [&](int p) {
    const Soa &s = soa[p]; float mx = -1e30f; int bi = -1;
    for (int i = 0; i < s.k; ++i) {
      if (s.skip[i]) continue;
      float u = -s.w[i] / s.n[i] + s.prob[i] * vs / (s.n[i] + 1);
      if (u > mx) { mx = u; bi = i; }
    }
    return bi; });
  run("parent arrays, 2-pass vector", [&](int p) {
    const Soa &s = soa[p]; float u[48];
    for (int i = 0; i < s.k; ++i)
      u[i] = s.skip[i] ? -1e30f : -s.w[i] / s.n[i] + s.prob[i] * vs / (s.n[i] + 1);
    float mx = -1e30f; for (int i = 0; i < s.k; ++i) mx = std::max(mx, u[i]);
    int bi = 0; while (u[bi] != mx) ++bi;
    return bi; });
}
