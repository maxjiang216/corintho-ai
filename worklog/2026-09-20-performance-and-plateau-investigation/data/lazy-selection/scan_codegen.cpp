#include <cstdint>
#include <utility>
struct Edge { uint16_t move_id : 7; uint16_t probability : 9; };
void v1(Edge *edges_, int first, int n) {
  int best = first;
  for (int i = first + 1; i < n; ++i) {
    const Edge e = edges_[i];
    const Edge b = edges_[best];
    if (e.probability > b.probability ||
        (e.probability == b.probability && e.move_id < b.move_id))
      best = i;
  }
  std::swap(edges_[first], edges_[best]);
}
void v2(Edge *edges_, int first, int n) {
  auto rank = [](Edge e) -> uint32_t {
    return (static_cast<uint32_t>(e.probability) << 7) | (127U - static_cast<uint32_t>(e.move_id)); };
  int best = first; uint32_t best_rank = rank(edges_[first]);
  for (int i = first + 1; i < n; ++i) { const uint32_t r = rank(edges_[i]); if (r > best_rank) { best_rank = r; best = i; } }
  std::swap(edges_[first], edges_[best]);
}
