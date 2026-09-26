#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
int main() {
  uint64_t n = 0, diff = 0, diff_ge1 = 0;
  float first = 0;
  for (uint32_t bits = 1;; ++bits) {
    float x; std::memcpy(&x, &bits, 4);
    if (x > 511.5f) break;
    ++n;
    long a = std::lround(x);
    int b = static_cast<int>(x + 0.5f);
    if (a != b) {
      if (!diff) first = x;
      ++diff;
      // after std::max(1, .) clamping, does the stored value differ?
      if ((a < 1 ? 1 : a) != (b < 1 ? 1 : b)) { ++diff_ge1; if (diff_ge1 <= 5) printf("  x=%.9g lround=%ld new=%d\n", x, a, b); }
    }
  }
  printf("floats checked %llu, raw diffs %llu (first %.9g), diffs after clamp %llu\n",
         (unsigned long long)n, (unsigned long long)diff, first, (unsigned long long)diff_ge1);
}
