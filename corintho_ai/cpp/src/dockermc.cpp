#include "dockermc.h"

#include <cstdint>

#include <memory>
#include <random>

#include "trainmc.h"
#include "util.h"

DockerMC::DockerMC(int32_t seed, int32_t max_searches,
                   int32_t searches_per_eval, float c_puct, float epsilon,
                   int32_t board[4 * kBoardSize], int32_t to_play,
                   int32_t pieces[6])
    : generator_(std::make_unique<std::mt19937>(seed)),
      to_eval_(std::make_unique<float[]>(searches_per_eval * kGameStateSize)),
      trainmc_(generator_.get(), to_eval_.get(), max_searches,
               searches_per_eval, c_puct, epsilon, board, to_play, pieces) {}

float DockerMC::eval() const noexcept {
  return trainmc_.eval();
}

int32_t DockerMC::num_requests() const noexcept {
  return trainmc_.num_requests();
}

int32_t DockerMC::num_nodes() const noexcept {
  return trainmc_.num_nodes();
}

bool DockerMC::done() const noexcept {
  return trainmc_.done();
}

bool DockerMC::drawn() const noexcept {
  return trainmc_.drawn();
}

void DockerMC::writeRequests(float *game_states) const noexcept {
  trainmc_.writeRequests(game_states);
}

void DockerMC::getLegalMoves(int32_t legal_moves[kNumMoves]) const noexcept {
  trainmc_.getLegalMoves(legal_moves);
}

int32_t DockerMC::chooseMove() noexcept {
  return trainmc_.chooseMove();
}

bool DockerMC::doIteration(float eval[], float probs[]) {
  return trainmc_.doIteration(eval, probs);
}