#include "trainer.h"

#include <cstdint>

#include <algorithm>
#include <queue>
#include <string>
#include <vector>

#include <gsl/gsl>
#include <omp.h>

#include "node.h"
#include "selfplayer.h"
#include "trainmc.h"
#include "util.h"

Trainer::Trainer(int32_t num_games, const std::string &log_folder,
                 int32_t seed, int32_t max_searches, int32_t searches_per_eval,
                 float c_puct, float epsilon, int32_t num_logged,
                 int32_t num_threads, bool testing)
    : is_done_{std::vector<uint8_t>(num_games, 0)},
      max_searches_{max_searches}, searches_per_eval_{searches_per_eval},
      num_threads_{num_threads}, generator_{gsl::narrow_cast<uint32_t>(seed)} {
  assert(num_games > 0);
  assert(num_logged >= 0);
  assert(num_logged <= num_games);
  assert(max_searches > 0);
  assert(searches_per_eval > 0);
  assert(max_searches >= searches_per_eval);
  assert(c_puct > 0.0);
  assert(epsilon >= 0.0);
  assert(epsilon <= 1.0);
  assert(num_threads > 0);
  initialize(num_games, log_folder, max_searches, searches_per_eval, c_puct,
             epsilon, num_logged, testing);
}

float *Trainer::slotRows(int32_t s) noexcept {
  return batch_.data() +
         static_cast<size_t>(s) * searches_per_eval_ * kGameStateSize;
}

const float *Trainer::requests() const noexcept {
  assert(!batch_.empty());
  return batch_.data();
}

int32_t Trainer::num_requests(int32_t to_play) const noexcept {
  // Training: whole slots, see batch_
  if (to_play != 0 && to_play != 1)
    return all_done_ ? 0 : searches_per_eval_ * num_active_;
  int32_t num_requests = 0;
  for (const auto &game : games_) {
    if (!is_done_[&game - &games_[0]] &&
        ((to_play != 0 && to_play != 1) ||
         game.to_play() == (to_play + game.parity()) % 2)) {
      num_requests += game.num_requests();
    }
  }
  return num_requests;
}

int32_t Trainer::num_samples() const noexcept {
  int32_t num_samples = 0;
  for (const auto &game : games_) {
    num_samples += game.num_samples();
  }
  return num_samples;
}

float Trainer::score() const noexcept {
  float score = 0;
  for (size_t i = 0; i < games_.size(); i += 2) {
    score += games_[i].score();
  }
  for (size_t i = 1; i < games_.size(); i += 2) {
    score += 1.0 - games_[i].score();
  }
  return score / games_.size();
}

int32_t Trainer::numWins() const noexcept {
  int32_t wins = 0;
  for (size_t i = 0; i < games_.size(); ++i) {
    const float s =
        (i % 2 == 0) ? games_[i].score() : 1.0F - games_[i].score();
    if (s == 1.0F)
      ++wins;
  }
  return wins;
}

int32_t Trainer::numDraws() const noexcept {
  int32_t draws = 0;
  for (size_t i = 0; i < games_.size(); ++i) {
    const float s =
        (i % 2 == 0) ? games_[i].score() : 1.0F - games_[i].score();
    if (s == 0.5F)
      ++draws;
  }
  return draws;
}

int32_t Trainer::numGames() const noexcept {
  return gsl::narrow_cast<int32_t>(games_.size());
}

float Trainer::avg_mate_length() const noexcept {
  int32_t total_length = 0;
  for (const auto &game : games_) {
    assert(is_done_[&game - &games_[0]]);
    total_length += game.mate_length();
  }
  return static_cast<float>(total_length) / games_.size();
}

void Trainer::writeRequests(float *game_states,
                            int32_t to_play) const noexcept {
  int32_t offset = 0;
  // Testing mode
  // Only count requests from one player
  if (to_play == 0 || to_play == 1) {
    for (size_t i = 0; i < games_.size(); ++i) {
      if (games_[i].to_play() == (to_play + games_[i].parity()) % 2 &&
          !is_done_[i]) {
        games_[i].writeRequests(game_states + offset * kGameStateSize);
        offset += games_[i].num_requests();
      }
    }
    return;
  }
  // Training mode: the slots are already a contiguous batch
  const size_t count =
      static_cast<size_t>(num_requests(to_play)) * kGameStateSize;
  std::copy(batch_.data(), batch_.data() + count, game_states);
}

void Trainer::writeSamples(float *game_states, float *eval_samples,
                           float *prob_samples) const noexcept {
  int32_t offset = 0;
  for (size_t i = 0; i < games_.size(); ++i) {
    games_[i].writeSamples(game_states +
                               offset * kGameStateSize * kNumSymmetries,
                           eval_samples + offset * kNumSymmetries,
                           prob_samples + offset * kNumMoves * kNumSymmetries);
    offset += games_[i].num_samples();
  }
}

void Trainer::writeScores(const std::string &filename) const {
  float scores[games_.size()];
  for (size_t i = 0; i < games_.size(); i += 2) {
    scores[i] = games_[i].score();
  }
  for (size_t i = 1; i < games_.size(); i += 2) {
    scores[i] = 1.0 - games_[i].score();
  }
  // First player score
  int32_t wins = 0;
  int32_t draws = 0;
  for (size_t i = 0; i < games_.size(); i += 2) {
    if (scores[i] == 1.0) {
      ++wins;
    } else if (scores[i] == 0.5) {
      ++draws;
    }
  }
  std::ofstream file = std::ofstream{filename, std::ofstream::out};
  file << "First player wins: " << wins << " / " << games_.size() / 2 << " = "
       << static_cast<float>(wins) / (games_.size() / 2)
       << "\nFirst player draws: " << draws << " / " << games_.size() / 2
       << " = " << static_cast<float>(draws) / (games_.size() / 2)
       << "\nFirst player losses: " << games_.size() / 2 - wins - draws
       << " / " << games_.size() / 2 << " = "
       << static_cast<float>(games_.size() / 2 - wins - draws) /
              (games_.size() / 2)
       << '\n';
  // Second player score
  wins = 0;
  draws = 0;
  for (size_t i = 1; i < games_.size(); i += 2) {
    if (scores[i] == 1.0) {
      ++wins;
    } else if (scores[i] == 0.5) {
      ++draws;
    }
  }
  file << "Second player wins: " << wins << " / " << games_.size() / 2 << " = "
       << static_cast<float>(wins) / (games_.size() / 2)
       << "\nSecond player draws: " << draws << " / " << games_.size() / 2
       << " = " << static_cast<float>(draws) / (games_.size() / 2)
       << "\nSecond player losses: " << games_.size() / 2 - wins - draws
       << " / " << games_.size() / 2 << " = "
       << static_cast<float>(games_.size() / 2 - wins - draws) /
              (games_.size() / 2)
       << '\n';
}

bool Trainer::doIteration(float eval[], float probs[], int32_t to_play) {
  // Training
  if (to_play != 0 && to_play != 1) {
    // We offset the start of the games to try to get an even distribution
    // of the games across the number of searches in a move. This way, the
    // total number of nodes will be more even. This reduces peak memory
    // usage. Avoid division by 0 in the rare case that games_.size() <
    // max_searches_. Games start in index order, each taking the next slot.
    //
    // searches_done_ counts iterations, and one turn is max_searches_ /
    // searches_per_eval_ of them (100 by default), so by default the starts
    // spread over max_searches_ iterations, ~16 turns, not one. That is the
    // stagger entry 19 measured (-30% peak memory). set_stagger_iterations()
    // chooses the span; the games themselves are the same either way.
    size_t due_count;
    if (stagger_iterations_ > 0) {
      due_count =
          std::min(games_.size(),
                   ((static_cast<size_t>(searches_done_) + 1) * games_.size() +
                    static_cast<size_t>(stagger_iterations_) - 1) /
                       static_cast<size_t>(stagger_iterations_));
    } else {
      const size_t stride =
          std::max(games_.size() / max_searches_, static_cast<size_t>(1));
      due_count = std::min(games_.size(),
                           (static_cast<size_t>(searches_done_) + 1) * stride);
    }
    const auto due = gsl::narrow_cast<int32_t>(due_count);
    for (; num_started_ < due; ++num_started_) {
      slot_of_[num_started_] = num_active_;
      game_in_slot_[num_active_] = num_started_;
      games_[num_started_].set_to_eval(slotRows(num_active_));
      ++num_active_;
    }
    // A game's results are in its slot, rows [slot * searches_per_eval_, ...)
    const size_t rows = static_cast<size_t>(searches_per_eval_);
    omp_set_num_threads(num_threads_);
    // Dynamic, one game at a time: a static schedule gives each thread a
    // contiguous range of games, and both the staggered starts above and the
    // hybrid CPU's slower E-cores then leave threads idle at the barrier.
    // Each game has its own RNG, so the schedule does not change the games.
#pragma omp parallel for schedule(dynamic, 1)
    for (size_t i = 0; i < games_.size(); ++i) {
      if (!is_done_[i]) {
        if (slot_of_[i] >= 0) {
          const size_t first = static_cast<size_t>(slot_of_[i]) * rows;
          // First search does not depend on pointers being null
          bool done =
              games_[i].doIteration(eval + first, probs + kNumMoves * first);
          // Game is done
          if (done) {
            is_done_[i] = true;
          }
        }
      }
    }
    ++searches_done_;
    // Keep the active slots a dense prefix: the game in the last active slot
    // moves into each finished game's slot, bringing the rows it has just
    // written. One small copy per finished game, not one per iteration.
    bool all_done = true;
    for (size_t i = 0; i < games_.size(); ++i) {
      if (!is_done_[i]) {
        all_done = false;
        continue;
      }
      const int32_t s = slot_of_[i];
      if (s < 0)
        continue;
      const int32_t last = num_active_ - 1;
      if (s != last) {
        const int32_t moved = game_in_slot_[last];
        std::copy(slotRows(last), slotRows(last) + rows * kGameStateSize,
                  slotRows(s));
        games_[moved].set_to_eval(slotRows(s));
        slot_of_[moved] = s;
        game_in_slot_[s] = moved;
      }
      slot_of_[i] = -1;
      --num_active_;
    }
    all_done_ = all_done;
    return all_done;
  }
  // Testing
  int32_t offset = 0;
  std::vector<int32_t> offsets(games_.size(), 0);
  for (size_t i = 1; i < games_.size(); ++i) {
    // Only count games from one player
    if (games_[i - 1].to_play() == (to_play + games_[i - 1].parity()) % 2 &&
        !is_done_[i - 1]) {
      offset += games_[i - 1].num_requests();
    }
    offsets[i] = offset;
  }
  omp_set_num_threads(num_threads_);
  // Dynamic for the same reason as in training (E-cores; games finish at
  // different times)
#pragma omp parallel for schedule(dynamic, 1)
  for (size_t i = 0; i < games_.size(); ++i) {
    // No offset in game start (there are not enough games for memory usage to
    // matter).
    if (games_[i].to_play() == (to_play + games_[i].parity()) % 2 &&
        !is_done_[i]) {
      bool done = games_[i].doIteration(eval + offsets[i],
                                        probs + kNumMoves * offsets[i]);
      if (done) {
        is_done_[i] = true;
      }
    }
  }
  for (size_t i = 0; i < games_.size(); ++i) {
    if (!is_done_[i]) {
      return false;
    }
  }
  return true;
}

void Trainer::initialize(int32_t num_games, const std::string &log_folder,
                         int32_t max_searches, int32_t searches_per_eval,
                         float c_puct, float epsilon, int32_t num_logged,
                         bool testing) {
  games_.reserve(num_games);
  // Training games write their network inputs into their own slot of one
  // shared batch (see batch_); testing games keep their own buffers.
  if (!testing) {
    batch_.assign(static_cast<size_t>(num_games) * searches_per_eval *
                      kGameStateSize,
                  0.0F);
    slot_of_.assign(num_games, -1);
    game_in_slot_.assign(num_games, -1);
  }
  auto slot = [&](int32_t i) -> float * {
    return testing ? nullptr
                   : batch_.data() + static_cast<size_t>(i) *
                                         searches_per_eval * kGameStateSize;
  };
  for (int32_t i = 0; i < num_logged; ++i) {
    games_.emplace_back(
        generator_(), max_searches, searches_per_eval, c_puct, epsilon,
        std::make_unique<std::ofstream>(log_folder + "/game_" +
                                            std::to_string(i) + ".txt",
                                        std::ofstream::out),
        testing, i % 2,  // Generate parity for test games (changes who plays
                         // first). Does not affect training games
        slot(i));
  }
  for (int32_t i = num_logged; i < num_games; ++i) {
    games_.emplace_back(generator_(), max_searches, searches_per_eval, c_puct,
                        epsilon, nullptr, testing, i % 2, slot(i));
  }
}

void Trainer::enableSolver(int32_t max_horizon, uint64_t max_nodes,
                           int32_t num_threads, int32_t log2_table) {
  solver_pool_ =
      std::make_unique<SolverPool>(num_threads, log2_table, max_nodes);
  setSolver(solver_pool_.get(), max_horizon);
}

void Trainer::setSolver(SolverPool *pool, int32_t max_horizon) {
  for (SelfPlayer &game : games_)
    game.set_solver(pool, max_horizon);
}

int32_t Trainer::numAdjudicated() const noexcept {
  int32_t n = 0;
  for (const SelfPlayer &game : games_)
    n += game.adjudicated();
  return n;
}
