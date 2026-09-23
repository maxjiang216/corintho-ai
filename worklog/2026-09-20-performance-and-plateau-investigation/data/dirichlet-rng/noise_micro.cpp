// Cost of one generateDirichlet call (the sampling loop only), three ways.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include "util.h"

__attribute__((noinline)) float old_way(std::mt19937 &g, int n, float *out) {
  float sum = 0;
  for (int i = 0; i < n; ++i) { out[i] = gamma_samples[g() % kNumGammaBuckets]; sum += out[i]; }
  return sum;
}
__attribute__((noinline)) float slice_mt(std::mt19937 &g, int n, float *out) {
  float sum = 0;
  for (int i = 0; i < n; i += 3) {
    uint32_t b = g(); int e = std::min(i + 3, n);
    for (int j = i; j < e; ++j) { out[j] = gamma_samples[b & 1023]; b >>= 10; sum += out[j]; }
  }
  return sum;
}
__attribute__((noinline)) float splitmix(uint64_t &s, int n, float *out) {
  float sum = 0;
  for (int i = 0; i < n; i += 6) {
    uint64_t z = (s += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    uint64_t b = z ^ (z >> 31); int e = std::min(i + 6, n);
    for (int j = i; j < e; ++j) { out[j] = gamma_samples[b & 1023]; b >>= 10; sum += out[j]; }
  }
  return sum;
}

static inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
struct Xo { uint64_t s[4]; };
__attribute__((noinline)) float xoshiro_ss(Xo &x, int n, float *out) {
  float sum = 0;
  for (int i = 0; i < n; i += 6) {
    uint64_t *s = x.s;
    uint64_t b = rotl(s[1] * 5, 7) * 9, t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t; s[3] = rotl(s[3], 45);
    int e = std::min(i + 6, n);
    for (int j = i; j < e; ++j) { out[j] = gamma_samples[b & 1023]; b >>= 10; sum += out[j]; }
  }
  return sum;
}
__attribute__((noinline)) float xoshiro_pp(Xo &x, int n, float *out) {
  float sum = 0;
  for (int i = 0; i < n; i += 6) {
    uint64_t *s = x.s;
    uint64_t b = rotl(s[0] + s[3], 23) + s[0], t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t; s[3] = rotl(s[3], 45);
    int e = std::min(i + 6, n);
    for (int j = i; j < e; ++j) { out[j] = gamma_samples[b & 1023]; b >>= 10; sum += out[j]; }
  }
  return sum;
}
int main() {
  // Legal-move counts drawn uniformly from 10..40 (mean ~25, as in the profile).
  std::mt19937 pick(1);
  const int kCalls = 2'000'000;
  std::vector<int> ns(kCalls);
  for (auto &n : ns) n = 10 + pick() % 31;
  float out[64]; volatile float sink = 0;
  auto time = [&](const char *name, auto fn) {
    double best = 1e9;
    for (int rep = 0; rep < 7; ++rep) {
      auto t0 = std::chrono::steady_clock::now();
      for (int n : ns) sink = sink + fn(n, out);
      double ns_per = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count() / kCalls;
      best = std::min(best, ns_per);
    }
    printf("%-10s %6.1f ns / call (min of 7)\n", name, best);
    return best;
  };
  std::mt19937 g(5); uint64_t s = 5;
  double a = time("old", [&](int n, float *o) { return old_way(g, n, o); });
  double b = time("slice mt", [&](int n, float *o) { return slice_mt(g, n, o); });
  double c = time("splitmix", [&](int n, float *o) { return splitmix(s, n, o); });
  Xo x{{1,2,3,4}};
  time("xoshiro**", [&](int n, float *o) { return xoshiro_ss(x, n, o); });
  time("xoshiro++", [&](int n, float *o) { return xoshiro_pp(x, n, o); });
  printf("slice %.1fx, splitmix %.1fx faster than old\n", a / b, a / c);
}
