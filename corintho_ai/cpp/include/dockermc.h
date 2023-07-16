#ifndef DOCKERMC_H
#define DOCKERMC_H

#include <memory>
#include <random>

#include "trainmc.h"

/// @brief A wrapper for TrainMC that is used in the web app
class DockerMC {
 public:
  /// @brief Constructor for web app
  DockerMC(int32_t max_searches, int32_t searches_per_eval, float c_puct,
           float epsilon, int32_t board[4 * kBoardSize], int32_t to_play,
           int32_t pieces[6]);
  ~DockerMC() = default;

  /// @brief Returns the evaluation of the root node
  float eval() const noexcept;
  /// @brief Returns the number of requests for evaluations
  int32_t num_requests() const noexcept;
  /// @brief Returns the number of nodes in the tree
  int32_t num_nodes() const noexcept;
  /// @brief Returns if the game is done.
  /// @details Returns if the root node is a terminal position.
  bool done() const noexcept;
  /// @brief Returns if the game is drawn.
  bool drawn() const noexcept;

  /// @brief Write the game states for the positions we need to evaluate.
  void writeRequests(float *game_states) const noexcept;
  /// @brief Get the legal moves of the root node.
  void getLegalMoves(int32_t legal_moves[kNumMoves]) const noexcept;

  /// @brief Find best move and move the root node to that node.
  /// @return The ID of the best move
  int32_t chooseMove() noexcept;
  /// @brief Do an iteration of searches
  bool doIteration(float eval[] = nullptr, float probs[] = nullptr);

 private:
  std::unique_ptr<std::mt19937> generator_;
  /// @brief The array to write the game states to.
  /// @details The wrapper is primarily to control the lifetime of this array.
  std::unique_ptr<float[]> to_eval_;
  TrainMC trainmc_;
};

#endif
