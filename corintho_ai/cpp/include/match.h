#ifndef MATCH_H
#define MATCH_H

#include <bitset>
#include <memory>

#include "game.h"
#include "trainmc.h"
#include "util.h"

/// @brief A tournament match with 2 players
class Match {
 public:
  Match(std::unique_ptr<TrainMC> player1, std::unique_ptr<TrainMC> player2,
        int32_t random_seed,
        std::unique_ptr<std::ofstream> log_file = nullptr);
  ~Match() = default;

  int32_t to_play() const noexcept;
  /// @brief Return the number of requests for evaluations
  /// @return The number of requests for evaluations
  int32_t num_requests() const noexcept;
  /// @brief The first player score
  /// @return The first player score
  /// @details 1.0 for a win, 0.0 for a loss, and 0.5 for a draw
  float score() const noexcept;

  /// @brief Write the game states for which evaluations are requested
  /// @param game_states The array to write the game states to
  /// This is the input to the neural network
  void writeRequests(float *game_states) const noexcept;

  /// @brief Do an iteration of searches for the current player
  /// @return If the game is complete
  bool doIteration(float eval[] = nullptr, float probs[] = nullptr);

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
  /// @brief Choose a move and write the training sample
  /// @return The ID of the chosen move
  int32_t chooseMove();
  /// @brief Choose a move and then do an iteration of searches
  bool chooseMoveAndContinue();
  /// @brief Random generator for all operations
  /// @details Shared with the TrainMC objects
  std::mt19937 generator_{};
  /// @brief Positions needing evaluation
  std::unique_ptr<float[]> to_eval_{};
  /// @brief Monte Carlo search trees for each player
  /// @details A nullptr indicates a random player
  std::unique_ptr<TrainMC> players_[2];
  /// @brief Current game state
  std::unique_ptr<Node> root_{};
  /// @brief Whose turn it is
  int32_t to_play_{0};
  /// @brief Game result for the first player
  Result result_{kResultNone};
  /// @brief File where all logs are written to
  /// @details We use a pointer so that no memory is allocated if there is no
  /// logging file (which is true most of the time).
  std::unique_ptr<std::ofstream> log_file_{};
};

#endif