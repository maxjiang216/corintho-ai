#include "tourney.h"

#include <random>

#include "util.h"

int32_t Tourney::num_requests(int32_t id) const noexcept {
  int32_t count = 0;
  for (const auto &match : matches_) {
    if (match.to_play() == id) {
      count += match.num_requests();
    }
  }
  return count;
}

void Tourney::writeRequests(float *game_states, int32_t id) noexcept {
  int32_t offset = 0;
  for (size_t i = 0; i < matches_.size(); ++i) {
    if (matches_[i].to_play() == id) {
      matches_[i].writeRequests(game_states + offset * kGameStateSize);
      offset += matches_[i].num_requests();
    }
  }
}

// feed eval and probs in order
// in pipeline, we will do
// for each id
// 1. write requests for id
// 2. doIteration for id
// so the next batch of write requests will be the next loop
bool Tourney::doIteration(float eval[], float probs[], int32_t id) {
  for (auto &match : matches_) {
    if (match.to_play() == id) {
      if (match.doIteration(eval, probs)) {
        return true;
      }
    }
  }
  return false;
}

void Tourney::addPlayer(int32_t player_id, int32_t model_id,
                        int32_t max_searches, int32_t searches_per_eval,
                        float c_puct, float epsilon, bool random) {
  players_[player_id] = {
      model_id, {max_searches, searches_per_eval, c_puct, epsilon, random}};
}

void Tourney::addMatch(int32_t player1, int32_t player2) {
  matches_.emplace_back(generator_(), players_[player1].params,
                        players_[player2].params);
}