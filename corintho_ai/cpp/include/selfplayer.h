#ifndef SELFPLAYER_H
#define SELFPLAYER_H

#include <cstdint>

#include <array>
#include <fstream>
#include <memory>
#include <random>
#include <utility>
#include <vector>

#include "solver_pool.h"
#include "trainmc.h"
#include "util.h"

/// @brief A training sample for the neural network
/// @details The evaluation sample is not store as it is computed only when the
/// game is complete.
struct Sample {
  Sample(std::array<float, kGameStateSize> game_state,
         std::array<float, kNumMoves> probabilities)
      : game_state{game_state}, probabilities{probabilities} {}
  std::array<float, kGameStateSize> game_state;
  std::array<float, kNumMoves> probabilities;
};

/// @brief A training match with 2 TrainMC players
class SelfPlayer {
 public:
  /// @brief Root of the tree now being searched (the player to move's),
  /// or nullptr. For measurements (worklog 2026-09-25-nn-architectures,
  /// entry 08).
  const Node *searchRoot() const noexcept { return players_[to_play_].root(); }
  SelfPlayer(int32_t random_seed, int32_t max_searches = 1600,
             int32_t searches_per_eval = 16, float c_puct = 1.0,
             float epsilon = 0.25,
             std::unique_ptr<std::ofstream> log_file = nullptr,
             bool testing = false, int32_t parity = 0,
             float *to_eval = nullptr);
  SelfPlayer(SelfPlayer &&other) = default;
  ~SelfPlayer() = default;

  int32_t to_play() const noexcept;
  int32_t parity() const noexcept;
  /// @brief Return the number of requests for evaluations
  /// @return The number of requests for evaluations
  int32_t num_requests() const noexcept;
  /// @brief Return the number of training samples
  /// @return The number of training samples
  int32_t num_samples() const noexcept;
  /// @brief The first player score
  /// @return The first player score
  /// @details 1.0 for a win, 0.0 for a loss, and 0.5 for a draw
  float score() const noexcept;
  /// @brief The length of the longest mating sequence found by a player
  /// @details Number of plies in the mating sequence. Also includes drawing
  /// sequences in a drawn position.
  int32_t mate_length() const noexcept;

  /// @brief Write the game states for which evaluations are requested
  /// @param game_states The array to write the game states to
  /// This is the input to the neural network
  void writeRequests(float *game_states) const noexcept;
  /// @brief Write the training samples
  /// @details This includes the game state, game outcome, and final move
  /// probabilities
  void writeSamples(float *game_states, float *eval_samples,
                    float *prob_samples) const noexcept;

  /// @brief Move this game's network inputs to another buffer
  /// @details The caller copies any pending rows first. Used by Trainer to
  /// keep the slots of active games a dense prefix of its batch.
  void set_to_eval(float *to_eval) noexcept;
  /// @brief Do an iteration of searches for the current player
  /// @return If the game is complete
  bool doIteration(float eval[] = nullptr, float probs[] = nullptr);

  /// @brief End games by exact solution once the position's horizon P is at
  /// most max_horizon (worklog 2026-09-25-nn-architectures, entry 14)
  /// @details The position is submitted to `pool` and the game ends at once
  /// (its trees and slot are freed); finalize() collects the result.
  void set_solver(SolverPool *pool, int32_t max_horizon) noexcept {
    solver_pool_ = pool;
    solve_horizon_ = max_horizon;
  }
  /// @brief Solve search leaves with horizon P <= max_horizon (entry 15)
  void set_node_solver(int32_t max_horizon, uint64_t max_nodes) noexcept {
    players_[0].set_node_solver(max_horizon, max_nodes);
    players_[1].set_node_solver(max_horizon, max_nodes);
  }
  /// @brief Whether the game ended by an exact solution
  bool adjudicated() const noexcept { return adjudicated_; }
  /// @brief Wait for a submitted solve and apply its result; call before
  /// writeSamples, score and the result counts. A capped solve is retried
  /// once with 20x the nodes; still unknown, the game counts as a draw.
  /// @return false if the result stayed unknown
  bool finalize();

 private:
  /// @brief Write the evaluation of the given node
  void writeEval(Node *node) const noexcept;
  /// @brief Write the legal movesof the root
  /// @details Also write entire main line and sort other moves by visits
  void writeMoves() const noexcept;
  /// @brief Logs information immediately before a move is chosen
  void writePreMoveLogs() const noexcept;
  /// @brief Log information about move choice immediately after it is done
  void writeMoveChoice(int32_t choice) const noexcept;
  /// @brief Clean up when game is complete
  void endGame() noexcept;
  /// @brief End the game with the exact result `value` (1, 0, -1 for the
  /// side to move at the current position)
  void adjudicate(int32_t value) noexcept;
  /// @brief At the start of an iteration: true when the position reached
  /// the solve horizon, was submitted, and the game has ended
  bool solveStep();
  /// @brief Choose a move and write the training sample
  /// @return The ID of the chosen move
  int32_t chooseMove();
  /// @brief Choose a move and then do an iteration of searches
  bool chooseMoveAndContinue();
  /// @brief Random generator for all operations
  /// @details Shared with the TrainMC objects
  std::mt19937 generator_{};
  /// @brief Buffer owned by this game when none was passed in
  std::unique_ptr<float[]> owned_to_eval_{};
  /// @brief Positions needing evaluation, searches_per_eval rows
  /// @details Either owned_to_eval_ or a slot in a buffer the caller owns
  /// (Trainer's batch, in training), in which case the rows are written
  /// straight into the network input and never copied.
  float *to_eval_{nullptr};
  /// @brief Monte Carlo search trees for each player
  TrainMC players_[2];
  /// @brief Whose turn it is
  int32_t to_play_{0};
  /// @brief Training samples
  std::vector<Sample> samples_{};
  /// @brief Game result for the first player
  Result result_{kResultNone};
  /// @brief Value target of the last sample (its player's result): 1 when
  /// the last mover won, 0 for a draw, -1 when an adjudicated game was lost
  float last_mover_value_{1.0F};
  SolverPool *solver_pool_{nullptr};
  int32_t solve_horizon_{0};
  std::shared_ptr<SolveJob> solve_job_{};

  bool adjudicated_{false};
  /// @brief File where all logs are written to
  /// @details We use a pointer so that no memory is allocated if there is no
  /// logging file (which is true most of the time).
  std::unique_ptr<std::ofstream> log_file_{};
  /// @brief Number of the turn at which some player finds a mating sequence
  /// (or draw)
  /// TODO: Test this, as it is currently very inaccurate.
  int32_t mate_turn_{0};
  /// @brief Parity for which model corresponds to which player during testing
  int32_t parity_{0};
  /// @brief If we are testing
  /// @details If we are testing, we do not write training samples
  bool testing_{false};
};

#endif