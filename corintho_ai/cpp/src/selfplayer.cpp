#include "selfplayer.h"

#include <cassert>

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "move.h"
#include "node.h"
#include "trainer.h"
#include "util.h"

SelfPlayer::SelfPlayer(int32_t random_seed, int32_t max_searches,
                       int32_t searches_per_eval, float c_puct, float epsilon,
                       std::unique_ptr<std::ofstream> log_file, bool testing,
                       int32_t parity)
    : generator_{std::make_unique<std::mt19937>(random_seed)},
      to_eval_{std::make_unique<float[]>(kGameStateSize * max_searches)},
      players_{TrainMC{generator_.get(), to_eval_.get(), max_searches,
                       searches_per_eval, c_puct, epsilon, testing},
               TrainMC{generator_.get(), to_eval_.get(), max_searches,
                       searches_per_eval, c_puct, epsilon, testing}},

      log_file_{std::move(log_file)}, parity_{parity} {
  assert(max_searches > 1);
  assert(searches_per_eval > 0);
  assert(c_puct > 0.0);
  assert(epsilon >= 0.0 && epsilon <= 1.0);
  assert(parity == 0 || parity == 1);
  if (!testing) {
    samples_.reserve(32);
  }
}

int32_t SelfPlayer::to_play() const noexcept {
  return to_play_;
}

int32_t SelfPlayer::parity() const noexcept {
  return parity_;
}

int32_t SelfPlayer::numRequests() const noexcept {
  return players_[to_play_].numNodesSearched();
}

int32_t SelfPlayer::numSamples() const noexcept {
  return samples_.size();
}

float SelfPlayer::score() const noexcept {
  // There are more first player losses
  if (result_ == kResultLoss)
    return 0.0;
  if (result_ == kResultWin)
    return 1.0;
  return 0.5;
}

int32_t SelfPlayer::mateLength() const noexcept {
  // No mate found
  if (mate_turn_ == 0)
    return 0;
  return samples_.size() - mate_turn_ + 1;
}

void SelfPlayer::writeRequests(float *game_states) const noexcept {
  int32_t count = kGameStateSize * players_[to_play_].numNodesSearched();
  std::copy(to_eval_.get(), to_eval_.get() + count, game_states);
}

void SelfPlayer::writeSamples(float *game_states, float *eval_samples,
                              float *prob_samples) const noexcept {
  // The last player to play a move is the winner, except in a draw
  float evaluation = 1.0;
  if (result_ == kResultDraw) {
    evaluation = 0.0;
  }
  // Start from end of the game to get evaluations more easily
  int32_t offset = 0;
  for (int32_t i = samples_.size() - 1; i >= 0; --i) {
    // Apply symmetries
    // The first symmetry is the identity, which is a bit inefficient but
    // makes the code simpler
    for (int32_t k = 0; k < kNumSymmetries; ++k) {
      for (int32_t j = 0; j < 4 * kBoardSize; ++j) {
        *(game_states + offset * kGameStateSize * kNumSymmetries +
          (k + 1) * kGameStateSize + j) =
            samples_[i].game_state[space_symmetries[k][j / 4] * 4 + j % 4];
      }
      for (int32_t j = 4 * kBoardSize; j < kGameStateSize; ++j) {
        *(game_states + offset * kGameStateSize * kNumSymmetries +
          (k + 1) * kGameStateSize + j) = samples_[i].game_state[j];
      }
      *(eval_samples + offset * kNumSymmetries + (k + 1)) = evaluation;
      for (int32_t j = 0; j < kNumMoves; ++j) {
        *(prob_samples + offset * kNumMoves * kNumSymmetries +
          (k + 1) * kNumMoves + j) =
            samples_[i].probabilities[move_symmetries[k][j]];
      }
    }
    ++offset;
    evaluation *= -1.0;
  }
}

bool SelfPlayer::doIteration(float eval[], float probs[]) {
  bool done = players_[to_play_].doIteration(eval, probs);
  // If we have completed a turn, we can choose a move
  if (done)
    return chooseMoveAndContinue();
  // Oterhwise, the turn is not done so the game is not done
  return false;
}

void SelfPlayer::writeEval(Node *node) const noexcept {
  assert(node != nullptr);
  assert(log_file_ != nullptr);
  // There is a forced sequence
  if (node->result() != kResultNone) {
    *log_file_ << strResult(node->result());
    return;
  }
  *log_file_ << std::fixed << std::setprecision(6)
             << node->evaluation() / node->visits();
}

void SelfPlayer::writeMoves() const noexcept {
  assert(log_file_ != nullptr);
  *log_file_ << "LEGAL MOVES:\n";
  // Print main line
  players_[to_play_].root()->printMainLine(log_file_.get());
  *log_file_ << '\n';
  // Get and sort remaining legal moves by visit count and evaluation
  struct MoveData {
    int32_t visits;
    float evaluation;
    float probability;
    int32_t move;
    Node *node;
    MoveData(int32_t visits, float evaluation, float probability, int32_t move,
             Node *node)
        : visits{visits}, evaluation{evaluation},
          probability{probability}, move{move}, node{node} {}
  };
  std::vector<MoveData> moves;
  Node *cur = players_[to_play_].root()->first_child();
  uintf edge_index = 0;
  while (cur != nullptr) {
    if (cur->child_id() == players_[to_play_].root()->move_id(edge_index)) {
      moves.emplace_back(cur->visits(),
                         cur->evaluation() / (float)cur->visits(),
                         players_[to_play_].root()->probability(edge_index),
                         cur->child_id(), cur);
    }
    ++edge_index;
  }
  sort(moves.begin(), moves.end(),
       [](const MoveData &a, const MoveData &b) -> bool {
         if (a.visits != b.visits)
           return a.visits > b.visits;
         if (a.evaluation != b.evaluation)
           return a.evaluation > b.evaluation;
         if (a.probability != b.probability)
           return a.probability > b.probability;
         return a.move < b.move;
       });
  // The first move is already printed in the main line
  for (int32_t i = 1; i < moves.size(); ++i) {
    *log_file_ << Move{moves[i].move} << " V: " << moves[i].visits << " E: ";
    writeEval(moves[i].node);
    *log_file_ << " P: " << moves[i].probability << '\t';
  }
  *log_file_ << '\n';
}

void SelfPlayer::writePreMoveLogs() const noexcept {
  assert(log_file_ != nullptr);
  *log_file_ << "TURN "
             << static_cast<int32_t>(players_[to_play_].root()->depth())
             << "\nPLAYER " << static_cast<int32_t>(to_play_ + 1)
             << " TO PLAY\nVISITS: "
             << static_cast<int32_t>(players_[to_play_].root()->visits())
             << '\n';
  *log_file_ << "POSITION EVALUATION: ";
  writeEval(players_[to_play_].root());
  *log_file_ << '\n';
  writeMoves();
}

void SelfPlayer::writeMoveChoice(int32_t choice) const noexcept {
  assert(log_file_ != nullptr);
  *log_file_ << "CHOSE MOVE " << Move{choice} << "\nNEW POSITION:\n"
             << players_[to_play_].root()->game() << "\n\n";
}

void SelfPlayer::endGame() noexcept {
  assert(players_[to_play_].root()->terminal());
  // Set result
  if (players_[to_play_].root()->result() == kResultDraw) {
    result_ = kResultDraw;
    // Second player win (to_play is not updated yet so it is opposite)
  } else if (to_play_ == 1) {
    result_ = kResultLoss;
  } else {
    result_ = kResultWin;
  }
  // Log game result
  if (log_file_ != nullptr) {
    if (result_ == kResultDraw) {
      *log_file_ << "GAME IS DRAWN.\n";
    } else {
      *log_file_ << "PLAYER " << to_play_ + 1 << " WON!\n";
    }
  }
  // Delete the players
  // We cannot delete the SelfPlayer yet as it contains training samples
  // and results which will be collected at the end
  players_[0].null_root();
  players_[1].null_root();
  generator_.reset();
  to_eval_.reset();
  log_file_.reset();
}

int32_t SelfPlayer::chooseMove() {
  std::array<float, kGameStateSize> game_state;
  std::array<float, kNumMoves> prob_sample;
  int32_t choice =
      players_[to_play_].chooseMove(game_state.data(), prob_sample.data());
  samples_.emplace_back(game_state, prob_sample);
  return choice;
}

bool SelfPlayer::chooseMoveAndContinue() {
  bool need_eval = false;
  // Loop until we need an evaluation.
  // We could play many turns, for example when mating sequences are found by
  // both players
  while (!need_eval) {
    if (log_file_ != nullptr) {
      writePreMoveLogs();
    }
    // New mate found
    if (players_[to_play_].root()->known() && mate_turn_ != 0) {
      mate_turn_ = samples_.size();
    }
    int32_t choice = chooseMove();
    if (log_file_ != nullptr) {
      writeMoveChoice(choice);
    }
    // Check if the game is over
    if (players_[to_play_].root()->terminal()) {
      endGame();
      return true;
    }
    // Go to next player
    to_play_ = 1 - to_play_;
    // First time iterating the second player
    if (players_[to_play_].isUninitialized()) {
      players_[to_play_].createRoot(players_[1 - to_play_].root()->game(),
                                    players_[1 - to_play_].root()->depth());
      // This is always false as the root requires an evaluation
      return players_[to_play_].doIteration();
    }
    // It's possible that we need an evaluation for this
    // in the case that received move has not been searched
    need_eval = players_[to_play_].receiveOpponentMove(
        choice, players_[1 - to_play_].root()->get_game(),
        players_[1 - to_play_].root()->depth());
    if (!need_eval) {
      // Otherwise, we search again.
      // If no evaluation is needed, this player also did all its iterations
      // without needing evaluations, so we loop again.
      // This can happen if a mating sequence is found
      need_eval = !players_[to_play_].doIteration();
    }
  }
  return false;
}
