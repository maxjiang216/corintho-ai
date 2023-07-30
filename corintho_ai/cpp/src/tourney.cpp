#include "tourney.h"

#include <cassert>
#include <cstdint>

#include <fstream>
#include <random>
#include <string>

#include <gsl/gsl>
#include <omp.h>

#include "util.h"

#include <iostream>
using namespace std;

bool Tourney::all_done() const noexcept {
  for (const auto &done : is_done_) {
    if (!done)
      return false;
  }
  return true;
}

int32_t Tourney::num_requests(int32_t id) const noexcept {
  int32_t count = 0;
  for (size_t i = 0; i < matches_.size(); ++i) {
    cerr << "i: " << i << " is_done: " << is_done_[i] << " to_play: " << matches_[i]->to_play() << " id: " << id << endl;
    if (!is_done_[i] && matches_[i]->to_play() == id) {
      count += matches_[i]->num_requests();
    }
  }
  return count;
}

void Tourney::writeScores(const std::string &filename) const {
  std::ofstream file = std::ofstream{filename, std::ofstream::out};
  for (size_t i = 0; i < matches_.size(); ++i) {
    if (is_done_[i]) {
      file << matches_[i]->id(0) << ' ' << matches_[i]->id(1) << ' '
           << matches_[i]->score() << '\n';
    }
  }
}

void Tourney::writeRequests(float *game_states, int32_t id) noexcept {
  int32_t offset = 0;
  for (size_t i = 0; i < matches_.size(); ++i) {
    if (!is_done_[i] && matches_[i]->to_play() == id) {
      matches_[i]->writeRequests(game_states + offset * kGameStateSize);
      offset += matches_[i]->num_requests();
    }
  }
}

void Tourney::doIteration(float eval[], float probs[], int32_t id) {
  int32_t offset = 0;
  int32_t offsets[matches_.size()] = {0};
  for (size_t i = 1; i < matches_.size(); ++i) {
    if (!is_done_[i] && matches_[i]->to_play() == id) {
      offset += matches_[i - 1]->num_requests();
    }
    offsets[i] = offset;
  }
  omp_set_num_threads(num_threads_);
#pragma omp parallel for
  for (size_t i = 0; i < matches_.size(); ++i) {
    if (!is_done_[i] && matches_[i]->to_play() == id) {
      bool done = matches_[i]->doIteration(eval + offsets[i],
                                           probs + offsets[i] * kNumMoves);
      if (done)
        is_done_[i] = true;
    }
  }
}

void Tourney::addPlayer(int32_t player_id, int32_t model_id,
                        int32_t max_searches, int32_t searches_per_eval,
                        float c_puct, float epsilon, bool random) {
  players_[player_id] = Player{model_id, max_searches, searches_per_eval,
                               c_puct,   epsilon,      random};
}

void Tourney::addMatch(int32_t player1, int32_t player2, bool logging) {
  assert(matches_.size() == is_done_.size());
  assert(players_.find(player1) != players_.end());
  assert(players_.find(player2) != players_.end());
  // make sure the player exists
  matches_.emplace_back(new Match{
      gsl::narrow_cast<int32_t>(generator_()), players_[player1],
      players_[player2],
      logging ? std::make_unique<std::ofstream>(
                    log_folder_ + "match_" + std::to_string(player1) + "_" +
                    std::to_string(player2) + "_" +
                    std::to_string(matches_.size()) + ".txt")
              : nullptr});
  is_done_.push_back(false);
}