// Uniformity of splitmix64-sliced bucket indices: each of 6 slots, and
// adjacent slot pairs jointly (top 5 bits each, 1024 cells).
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>
static uint64_t s = 0x1234567887654321ULL;
static uint64_t next() {
  uint64_t z = (s += 0x9e3779b97f4a7c15ULL);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}
int main() {
  const int64_t draws = 10'000'000;
  std::vector<std::vector<int64_t>> slot(6, std::vector<int64_t>(1024)), pair(5, std::vector<int64_t>(1024));
  for (int64_t d = 0; d < draws; ++d) {
    uint64_t b = next(); int v[6];
    for (int k = 0; k < 6; ++k) { v[k] = b & 1023; b >>= 10; ++slot[k][v[k]]; }
    for (int k = 0; k < 5; ++k) ++pair[k][(v[k] >> 5) * 32 + (v[k + 1] >> 5)];
  }
  auto z = [&](const std::vector<int64_t> &c) {
    double e = double(draws) / 1024, x = 0;
    for (auto v : c) x += (v - e) * (v - e) / e;
    return (x - 1023) / std::sqrt(2 * 1023.0);
  };
  for (int k = 0; k < 6; ++k) printf("slot %d      z = %+.2f\n", k, z(slot[k]));
  for (int k = 0; k < 5; ++k) printf("pair %d-%d    z = %+.2f\n", k, k + 1, z(pair[k]));
}
