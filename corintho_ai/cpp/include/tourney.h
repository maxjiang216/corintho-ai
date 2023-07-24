#ifndef TOURNEY_H
#define TOURNEY_H

#include <cstdint>

#include <map>

#include "match.h"

struct Player {
  int32_t model_id;
  int32_t max_searches;
  int32_t searches_per_eval;
  float c_puct;
  float epsilon;
  bool random;
};

class Tourney {
 public:
  Tourney();
  ~Tourney() = default;

  /// @brief Return the number of requests for evaluations for a given model
  int32_t num_requests(int32_t id) const noexcept;

  /// @brief Write the game states for which evaluations are requested for a
  /// given model
  void writeRequests(float *game_states, int32_t id) noexcept;

  /// @brief Iterate the games until an evaluation is needed
  bool doIteration(float eval[], float probs[]);
  void addPlayer(int32_t player_id, int32_t model_id,
                 int32_t max_searches = 1600, int32_t searches_per_eval = 16,
                 float c_puct = 1.0, float epsilon = 0.25, bool random = false);
  void addMatch(int32_t player1, int32_t player2);

 private:
  std::vector<Match> matches_;
  std::map<int32_t, Player> players_;
};

#endif