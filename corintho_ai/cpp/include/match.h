#ifndef MATCH_H
#define MATCH_H

#include <cstdint>

#include <bitset>
#include <memory>
#include <string>

#include <spdlog/spdlog.h>

#include "game.h"
#include "node.h"
#include "trainmc.h"
#include "util.h"

struct Player {
  Player() = default;
  Player(int32_t player_id, int32_t model_id, int32_t max_searches,
         int32_t searches_per_eval, float c_puct, float epsilon,
         bool random = false)
      : player_id{player_id}, model_id{model_id}, max_searches{max_searches},
        searches_per_eval{searches_per_eval}, c_puct{c_puct}, epsilon{epsilon},
        random{random} {}
  int32_t player_id{};
  int32_t model_id{};
  int32_t max_searches{1600};
  int32_t searches_per_eval{16};
  float c_puct{1.0};
  float epsilon{0.25};
  bool random{false};
};

/// @brief A tournament match with 2 players
class Match {
 public:
  Match(int32_t random_seed, Player player1, Player player2,
        const std::string &log_file);
  Match(const Match &other) = delete;
  Match(Match &&other) noexcept = default;
  Match &operator=(const Match &other) = delete;
  Match &operator=(Match &&other) noexcept = default;
  ~Match() = default;

  int32_t id(int32_t i) const noexcept;
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

  void enableDebugLogging() noexcept;

 private:
  /// @brief Write the evaluation of the given node
  std::string writeEval(Node *node) const noexcept;
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
  /// TODO: Add IDs for players?
  std::unique_ptr<TrainMC> players_[2];
  /// @brief Player IDs
  std::array<int32_t, 2> ids_{};
  /// @brief Model IDs for the players
  std::array<int32_t, 2> model_ids_;
  /// @brief Current game state
  std::unique_ptr<Node> root_ = std::make_unique<Node>();
  /// @brief Whose turn it is
  int32_t to_play_{0};
  /// @brief Game result for the first player
  Result result_{kResultNone};
  /// @brief File where all logs are written to
  /// @details We use a pointer so that no memory is allocated if there is no
  /// logging file (which is true most of the time).
  std::shared_ptr<spdlog::logger> logger_{};
  std::shared_ptr<spdlog::logger> debug_logger_{};
};

#endif