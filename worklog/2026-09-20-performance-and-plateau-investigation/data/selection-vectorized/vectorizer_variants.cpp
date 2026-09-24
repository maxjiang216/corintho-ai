#include <cstdint>
#include <limits>
struct Edge { uint16_t move_id : 7; uint16_t probability : 9; };
const float kNegInf = -std::numeric_limits<float>::infinity();
// v1: as written in the engine
void v1(const Edge *e, float den, const float *ev, const float *vi, const uint8_t *fl, int n, float vs, float *score) {
  for (int i = 0; i < n; ++i) {
    const float weighted = static_cast<float>(e[i].probability) * den * vs;
    const float visits = vi[i];
    const float normal = -1.0 * ev[i] / visits + weighted / (visits + 1.0);
    const float u = (fl[i] & 2) ? weighted : normal;
    score[i] = (fl[i] & 1) ? kNegInf : u;
  }
}
// v2: probabilities pre-extracted to float, selects on float masks
void v2(const float *p, const float *ev, const float *vi, const uint8_t *fl, int n, float vs, float *score) {
  for (int i = 0; i < n; ++i) {
    const float weighted = p[i] * vs;
    const float visits = vi[i];
    const float normal = -1.0 * ev[i] / visits + weighted / (visits + 1.0);
    const int f = fl[i];
    float u = normal;
    u = (f & 2) != 0 ? weighted : u;
    u = (f & 1) != 0 ? kNegInf : u;
    score[i] = u;
  }
}
#include <cstring>
// v3: selects as integer bit masks
void v3(const float *p, const float *ev, const float *vi, const uint8_t *fl, int n, float vs, float *score) {
  for (int i = 0; i < n; ++i) {
    const float weighted = p[i] * vs;
    const float visits = vi[i];
    const float normal = -1.0 * ev[i] / visits + weighted / (visits + 1.0);
    uint32_t wb, nb, ib; const float ninf = kNegInf;
    std::memcpy(&wb, &weighted, 4); std::memcpy(&nb, &normal, 4); std::memcpy(&ib, &ninf, 4);
    const uint32_t drawn = 0u - ((fl[i] >> 1) & 1u);
    const uint32_t skip = 0u - (fl[i] & 1u);
    uint32_t ub = (wb & drawn) | (nb & ~drawn);
    ub = (ib & skip) | (ub & ~skip);
    std::memcpy(&score[i], &ub, 4);
  }
}
