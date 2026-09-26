// Variants for expanding 64 board bits into 64 floats (1.0f / 0.0f).
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <bitset>
#include <vector>
#include <chrono>
#include <immintrin.h>

static const int N = 64;

// A: what the engine does today.
static void vA(const std::bitset<64> &b, float *o) {
  for (int32_t i = 0; i < N; ++i) {
    if (b[i]) o[i] = 1.0; else o[i] = 0.0;
  }
}
// B: branchless scalar, let the compiler vectorize.
static void vB(uint64_t b, float *o) {
  for (int32_t i = 0; i < N; ++i) o[i] = static_cast<float>((b >> i) & 1);
}
// C: nibble table, 16 * 4 floats = 256 bytes.
alignas(64) static float kNib[16][4];
static void vC(uint64_t b, float *o) {
  for (int32_t i = 0; i < 16; ++i)
    std::memcpy(o + i * 4, kNib[(b >> (i * 4)) & 0xF], 16);
}
// D: byte table, 256 * 8 floats = 8 KB.
alignas(64) static float kByte[256][8];
static void vD(uint64_t b, float *o) {
  for (int32_t i = 0; i < 8; ++i)
    std::memcpy(o + i * 8, kByte[(b >> (i * 8)) & 0xFF], 32);
}
// E: SSE2 only -- what the shipped module (no -march) can actually use.
static void vE(uint64_t b, float *o) {
  const __m128i one = _mm_set1_epi32(1);
  const __m128 onef = _mm_set1_ps(1.0F);
  for (int32_t i = 0; i < 16; ++i) {
    uint32_t n = (b >> (i * 4)) & 0xF;
    __m128i v = _mm_set1_epi32(static_cast<int>(n));
    __m128i m = _mm_setr_epi32(1, 2, 4, 8);
    v = _mm_and_si128(v, m);
    v = _mm_cmpeq_epi32(v, m);
    _mm_storeu_ps(o + i * 4, _mm_and_ps(_mm_castsi128_ps(v), onef));
    (void)one;
  }
}
#if defined(__AVX2__)
// F: AVX2, 8 bits at a time.
static void vF(uint64_t b, float *o) {
  const __m256 onef = _mm256_set1_ps(1.0F);
  const __m256i m = _mm256_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128);
  for (int32_t i = 0; i < 8; ++i) {
    __m256i v = _mm256_set1_epi32(static_cast<int>((b >> (i * 8)) & 0xFF));
    v = _mm256_and_si256(v, m);
    v = _mm256_cmpeq_epi32(v, m);
    _mm256_storeu_ps(o + i * 8, _mm256_and_ps(_mm256_castsi256_ps(v), onef));
  }
}
#endif

int main() {
  for (int i = 0; i < 16; ++i) for (int j = 0; j < 4; ++j) kNib[i][j] = (i >> j) & 1;
  for (int i = 0; i < 256; ++i) for (int j = 0; j < 8; ++j) kByte[i][j] = (i >> j) & 1;

  // Varying inputs, as in the engine: every call sees a different board.
  const int P = 4096;
  std::vector<uint64_t> words(P);
  uint64_t s = 0x9E3779B97F4A7C15ULL;
  for (int i = 0; i < P; ++i) {
    s ^= s << 13; s ^= s >> 7; s ^= s << 17;
    words[i] = s;
  }
  std::vector<std::bitset<64>> bits(P);
  for (int i = 0; i < P; ++i) bits[i] = std::bitset<64>(words[i]);

  alignas(64) float out[80], ref[80];
  // correctness
  for (int i = 0; i < P; ++i) {
    vA(bits[i], ref);
    auto chk = [&](const char *n, void (*f)(uint64_t, float *)) {
      std::memset(out, 0xAB, sizeof out); f(words[i], out);
      if (std::memcmp(out, ref, N * sizeof(float)) != 0) { printf("MISMATCH %s at %d\n", n, i); exit(1); }
    };
    chk("B", vB); chk("C", vC); chk("D", vD); chk("E", vE);
#if defined(__AVX2__)
    chk("F", vF);
#endif
  }
  printf("all variants agree with the current implementation\n");

  const int R = 20000;
  volatile float sink = 0;
  auto bench = [&](const char *name, auto fn) {
    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < R; ++r)
      for (int i = 0; i < P; ++i) { fn(i, out); sink += out[0]; }
    auto t1 = std::chrono::steady_clock::now();
    double ns = std::chrono::duration<double, std::nano>(t1 - t0).count() / ((double)R * P);
    printf("  %-28s %6.2f ns/call\n", name, ns);
  };
  bench("A current (bitset[])", [&](int i, float *o) { vA(bits[i], o); });
  bench("B branchless scalar",  [&](int i, float *o) { vB(words[i], o); });
  bench("C nibble table 256B",  [&](int i, float *o) { vC(words[i], o); });
  bench("D byte table 8KB",     [&](int i, float *o) { vD(words[i], o); });
  bench("E SSE2 (baseline ISA)",[&](int i, float *o) { vE(words[i], o); });
#if defined(__AVX2__)
  bench("F AVX2",               [&](int i, float *o) { vF(words[i], o); });
#endif
  printf("sink %f\n", (float)sink);
  return 0;
}
