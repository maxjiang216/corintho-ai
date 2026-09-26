#ifndef TRAINER_H
#define TRAINER_H

#include <cstdint>

#include <memory>
#include <random>
#include <string>
#include <vector>

#include "selfplayer.h"
#include "util.h"

/// @brief Orchestrates many SelfPlayer objects to generate training samples
/// from self-play games
/// @details This is the class that is used by Cython
class Trainer {
 public:
  /// @brief Default constructor
  /// This is needed for Cython to be able to create a Trainer object
  Trainer() = default;
  Trainer(int32_t num_games, const std::string &log_folder, int32_t seed,
          int32_t max_searches = 1600, int32_t searches_per_eval = 16,
          float c_puct = 1.0, float epsilon = 0.25, int32_t num_logged = 10,
          int32_t num_threads = 1, bool testing = false);
  ~Trainer() = default;

  /// @brief Return the number of requests for evaluations
  /// @details In training (to_play -1) this is the number of batch rows to
  /// evaluate: searches_per_eval rows for every active game, including the
  /// few rows a game left unused, whose outputs are ignored. 0 once every
  /// game is done.
  int32_t num_requests(int32_t to_play = -1) const noexcept;
  /// @brief The training batch, num_requests(-1) rows, to evaluate in place
  /// @details Each active game owns one slot of searches_per_eval rows and
  /// writes them during search, so nothing is copied. Results go back to
  /// doIteration in the same row layout. Training only.
  const float *requests() const noexcept;
  /// @brief The game whose rows are in batch slot `slot` (training only)
  /// @details Rows [slot * searches_per_eval, (slot + 1) * searches_per_eval)
  /// of requests() belong to this game. For measurements (worklog
  /// 2026-09-25-nn-architectures, entry 07).
  int32_t gameInSlot(int32_t slot) const noexcept {
    return game_in_slot_[static_cast<size_t>(slot)];
  }
  /// @brief Return the number of training samples
  int32_t num_samples() const noexcept;
  /// @brief Average score of first player
  float score() const noexcept;
  /// @brief Number of test games the new agent won outright
  /// @details Same parity handling as score(): even-indexed games count the
  /// first player's result and odd-indexed games the second player's, which is
  /// how the colours are balanced across the match. Used by the promotion
  /// gate, which tests the decisive-game win rate and so needs counts rather
  /// than an averaged score.
  int32_t numWins() const noexcept;
  /// @brief Number of drawn test games
  int32_t numDraws() const noexcept;
  /// @brief Number of games in this Trainer
  int32_t numGames() const noexcept;
  /// @brief Return the average mate length
  float avg_mate_length() const noexcept;

  /// @brief Write the game states for which evaluations are requested
  /// @param game_states The array to write the game states to
  /// @details In training this copies requests(); prefer evaluating that
  /// buffer in place.
  void writeRequests(float *game_states, int32_t to_play = -1) const noexcept;
  /// @brief Write the training samples
  void writeSamples(float *game_states, float *eval_samples,
                    float *prob_samples) const noexcept;
  /// @brief Write a summary of the game outcomes to a file
  /// @param file The name of the file to write to
  /// @return The percentage score of the first player. This is used to
  /// determine if a generation improved
  void writeScores(const std::string &file) const;

  /// @brief Spread the training games' starts evenly over this many
  /// iterations. 0 (the default) keeps the original rule, one start every
  /// max_searches / num_games iterations, which spreads them over
  /// max_searches iterations (~16 turns at 16 searches per iteration).
  void set_stagger_iterations(int32_t iterations) noexcept {
    stagger_iterations_ = iterations;
  }

  /// @brief This is the main function that runs the self-play games. It is
  /// called by Cython in a loop.
  /// @return If all games are done
  bool doIteration(float eval[], float probs[], int32_t to_play = -1);

 private:
  // Initialize SelfPlayers (factored out of different version of constructor)
  void initialize(int32_t num_games, const std::string &log_folder,
                  int32_t max_searches, int32_t searches_per_eval,
                  float c_puct, float epsilon, int32_t num_logged,
                  bool testing);

  /// @brief Start of slot s in batch_
  float *slotRows(int32_t s) noexcept;

  /// @brief The training batch: one slot of searches_per_eval rows per game
  /// @details Each game's search writes its network inputs straight into its
  /// slot. Almost every slot is full at every iteration (15.91 of 16 rows on
  /// average, worklog entry 23), so evaluating whole slots wastes ~0.6% of
  /// rows, whereas packing them was a serial copy costing ~16% of engine time
  /// at 20 threads. The active games' slots are kept a dense prefix: a game
  /// takes the next slot when it starts, and when it finishes, the game in the
  /// last active slot moves into its place (without that, finished games left
  /// ~48% of rows empty in a 1000-game run). Empty in testing, where each call
  /// serves only the games of one model and the requests are packed instead.
  std::vector<float> batch_{};
  /// @brief Slot of each game, -1 before it starts and after it finishes
  std::vector<int32_t> slot_of_{};
  /// @brief Game in each active slot
  std::vector<int32_t> game_in_slot_{};
  /// @brief Number of active slots, the prefix of batch_ to evaluate
  int32_t num_active_{0};
  /// @brief Number of games started so far; games start in index order
  int32_t num_started_{0};
  /// @brief The self-play games
  std::vector<SelfPlayer> games_{};
  /// @brief Tracks which games are done
  // uint8_t, not bool: std::vector<bool> packs 64 flags per word, so the
  // parallel writes in doIteration() are read-modify-writes of a shared word
  // and can drop each other's updates. See worklog 2026-09-21.
  std::vector<uint8_t> is_done_{};
  /// @brief Set once every game is done
  bool all_done_{false};
  /// @brief Maximum number of searches per turn for the players
  /// @details This is used to compute offsets for starting the games
  int32_t max_searches_{1600};
  /// @brief The maximum number of searches per neural network evaluation
  /// @details This is used to compute the size of arrays for neural network
  /// input and output
  int32_t searches_per_eval_{16};
  /// @brief The number of threads to use
  /// @details If 0, then use the number of threads available (which OpenMP
  /// will automatically do)
  int32_t num_threads_{0};
  /// @brief The number of searches done so far
  /// TODO: Is this searches or evaluations?
  int32_t searches_done_{0};
  /// @brief Span of the staggered start, in iterations; 0 for the original
  std::int32_t stagger_iterations_{0};
  /// @brief Random number generator
  std::mt19937 generator_{};
};

#endif