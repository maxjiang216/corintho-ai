#ifndef TOURNEY_H
#define TOURNEY_H

#include <cstdint>

#include <map>
#include <memory>
#include <random>
#include <string>

#include "match.h"

class Tourney {
 public:
  Tourney(int32_t num_threads, std::string log_folder)
      : num_threads_{num_threads}, log_folder_{log_folder} {
    assert(num_threads > 0);
  }
  ~Tourney() = default;

  /// @brief Return true if all games are done
  bool all_done() const noexcept;
  /// @brief Return the number of requests for evaluations for a given model
  int32_t num_requests(int32_t id) const noexcept;

  /// @brief Write completed game results into a file
  void writeScores(const std::string &filename) const;
  /// @brief Write the game states for which evaluations are requested for a
  /// given model
  void writeRequests(float *game_states, int32_t id) noexcept;

  /// @brief Iterate the games until an evaluation is needed
  void doIteration(float eval[], float probs[], int32_t id);
  void addPlayer(int32_t player_id, int32_t model_id,
                 int32_t max_searches = 1600, int32_t searches_per_eval = 16,
                 float c_puct = 1.0, float epsilon = 0.25,
                 bool random = false);
  void addMatch(int32_t player1, int32_t player2, bool logging = false);

 private:
  std::vector<std::unique_ptr<Match>> matches_{};
  std::vector<bool> is_done_{};
  std::map<int32_t, Player> players_{};
  std::mt19937 generator_{};
  int32_t num_threads_{1};
  std::string log_folder_{};
};

#endif