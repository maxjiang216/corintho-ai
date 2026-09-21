// Microbenchmarks for the game simulation layer.
//
// These are the functions PLAN.md section 13 proposes to rewrite as bitboard
// operations. Every one of them is called from the Node constructor, so they
// run once per node created during search.
//
// The corpus is built by random legal playouts from the starting position, so
// the distribution of positions resembles what search actually visits.

#include <cstdint>

#include <bitset>
#include <cstdio>
#include <vector>

#include "bench_util.h"
#include "game.h"
#include "node.h"
#include "util.h"

namespace {

/// @brief Build a corpus of positions by random legal play.
std::vector<Game> buildCorpus(size_t target, uint64_t seed) {
  std::vector<Game> corpus;
  corpus.reserve(target);
  uint64_t rng = seed;
  std::bitset<kNumMoves> legal;
  while (corpus.size() < target) {
    Game game;
    // Play a random game, recording every position along the way.
    while (corpus.size() < target) {
      corpus.push_back(game);
      game.getLegalMoves(legal);
      const size_t count = legal.count();
      if (count == 0)
        break;  // Terminal position; start a new game.
      size_t pick = bench::splitmix64(rng) % count;
      int32_t move_id = 0;
      for (int32_t i = 0; i < kNumMoves; ++i) {
        if (legal[i]) {
          if (pick == 0) {
            move_id = i;
            break;
          }
          --pick;
        }
      }
      game.doMove(move_id);
    }
  }
  return corpus;
}

/// @brief Distribution of legal move counts. Sizes the edge arrays and tells us
/// how much of the 96-move space is actually live.
void reportCorpus(const std::vector<Game> &corpus) {
  std::vector<int32_t> histogram(kNumMoves + 1, 0);
  std::bitset<kNumMoves> legal;
  uint64_t total = 0;
  int32_t terminal = 0;
  for (const Game &game : corpus) {
    game.getLegalMoves(legal);
    const int32_t count = static_cast<int32_t>(legal.count());
    ++histogram[count];
    total += count;
    if (count == 0)
      ++terminal;
  }
  std::printf("  positions %zu   mean legal moves %.2f   terminal %d (%.2f%%)\n",
              corpus.size(), static_cast<double>(total) / corpus.size(),
              terminal, 100.0 * terminal / corpus.size());
  int32_t lo = kNumMoves;
  int32_t hi = 0;
  for (int32_t i = 0; i <= kNumMoves; ++i) {
    if (histogram[i] > 0) {
      if (i < lo)
        lo = i;
      if (i > hi)
        hi = i;
    }
  }
  std::printf("  legal move count range [%d, %d]\n", lo, hi);
}

}  // namespace

int main(int argc, char **argv) {
  const size_t corpus_size = argc > 1 ? std::stoul(argv[1]) : 20000;
  const int32_t reps = argc > 2 ? std::stoi(argv[2]) : 50;

  std::printf("Corintho microbenchmarks\n");
  std::printf("corpus %zu positions, %d repetitions\n\n", corpus_size, reps);

  std::vector<Game> corpus = buildCorpus(corpus_size, 0x1234ABCDULL);
  reportCorpus(corpus);
  std::printf("\n");

  // --- getLegalMoves: the hot path, called once per node ---
  {
    std::bitset<kNumMoves> legal;
    auto start = bench::Clock::now();
    for (int32_t r = 0; r < reps; ++r) {
      for (const Game &game : corpus) {
        bool lines = game.getLegalMoves(legal);
        bench::keep(lines);
        bench::keep(legal);
      }
    }
    bench::Result{"Game::getLegalMoves", bench::secondsSince(start),
                  static_cast<uint64_t>(reps) * corpus.size()}
        .report();
  }

  // --- writeGameState: once per node that requests an evaluation ---
  {
    float state[kGameStateSize];
    auto start = bench::Clock::now();
    for (int32_t r = 0; r < reps; ++r) {
      for (const Game &game : corpus) {
        game.writeGameState(state);
        bench::keep(state);
      }
    }
    bench::Result{"Game::writeGameState", bench::secondsSince(start),
                  static_cast<uint64_t>(reps) * corpus.size()}
        .report();
  }

  // --- doMove: once per node, inside the Node constructor ---
  {
    // The original version of this benchmark took the LOWEST-indexed legal
    // move. Move IDs 0-47 are all move-moves and 48-95 are all places, so that
    // picked a move-move whenever one existed -- the expensive branch, the same
    // one every repetition, perfectly predicted. It also folded the Game copy
    // into the reported figure.
    //
    // Here: a uniformly chosen legal move per position, split into the place
    // and move-move branches and reported separately, with the copy timed on
    // its own so it can be subtracted.
    std::vector<Game> playable;
    std::vector<int32_t> any_move, place_move, tower_move;
    std::vector<Game> place_pos, tower_pos;
    uint64_t rng = 0x9E3779B97F4A7C15ULL;
    std::bitset<kNumMoves> legal;
    for (const Game &game : corpus) {
      game.getLegalMoves(legal);
      if (legal.count() == 0)
        continue;
      std::vector<int32_t> ids;
      for (int32_t i = 0; i < kNumMoves; ++i)
        if (legal[i])
          ids.push_back(i);
      playable.push_back(game);
      any_move.push_back(ids[bench::splitmix64(rng) % ids.size()]);
      // Separate pools so each branch is measured without the other's
      // mispredictions mixed in.
      std::vector<int32_t> places, towers;
      for (int32_t id : ids)
        (id >= 48 ? places : towers).push_back(id);
      if (!places.empty()) {
        place_pos.push_back(game);
        place_move.push_back(places[bench::splitmix64(rng) % places.size()]);
      }
      if (!towers.empty()) {
        tower_pos.push_back(game);
        tower_move.push_back(towers[bench::splitmix64(rng) % towers.size()]);
      }
    }

    // Copy alone, to subtract from the combined figures.
    auto start = bench::Clock::now();
    for (int32_t r = 0; r < reps; ++r) {
      for (size_t i = 0; i < playable.size(); ++i) {
        Game copy = playable[i];
        bench::keep(copy);
      }
    }
    bench::Result{"Game copy (alone)", bench::secondsSince(start),
                  static_cast<uint64_t>(reps) * playable.size()}
        .report();

    // Kept for comparability with earlier recorded runs, but now with a
    // uniformly chosen move rather than the lowest-indexed one.
    start = bench::Clock::now();
    for (int32_t r = 0; r < reps; ++r) {
      for (size_t i = 0; i < playable.size(); ++i) {
        Game copy = playable[i];
        copy.doMove(any_move[i]);
        bench::keep(copy);
      }
    }
    bench::Result{"Game::doMove (incl. copy)", bench::secondsSince(start),
                  static_cast<uint64_t>(reps) * playable.size()}
        .report();

    start = bench::Clock::now();
    for (int32_t r = 0; r < reps; ++r) {
      for (size_t i = 0; i < place_pos.size(); ++i) {
        Game copy = place_pos[i];
        copy.doMove(place_move[i]);
        bench::keep(copy);
      }
    }
    bench::Result{"Game::doMove place (incl. copy)", bench::secondsSince(start),
                  static_cast<uint64_t>(reps) * place_pos.size()}
        .report();

    start = bench::Clock::now();
    for (int32_t r = 0; r < reps; ++r) {
      for (size_t i = 0; i < tower_pos.size(); ++i) {
        Game copy = tower_pos[i];
        copy.doMove(tower_move[i]);
        bench::keep(copy);
      }
    }
    bench::Result{"Game::doMove move-move (incl. copy)",
                  bench::secondsSince(start),
                  static_cast<uint64_t>(reps) * tower_pos.size()}
        .report();
  }

  // --- Node construction: allocation + getLegalMoves + edge array fill ---
  {
    const int32_t node_reps = reps / 5 > 0 ? reps / 5 : 1;
    auto start = bench::Clock::now();
    for (int32_t r = 0; r < node_reps; ++r) {
      for (const Game &game : corpus) {
        Node *node = new Node(game, 0);
        bench::keep(node);
        delete node;
      }
    }
    bench::Result{"Node ctor+dtor (alloc heavy)", bench::secondsSince(start),
                  static_cast<uint64_t>(node_reps) * corpus.size()}
        .report();
  }

  std::printf("\n");
  return 0;
}
