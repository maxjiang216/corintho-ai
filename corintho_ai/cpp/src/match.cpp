#include "match.h"

#include <cassert>

#include <fstream>
#include <iomanip>
#include <memory>
#include <random>
#include <vector>

#include "node.h"

Match::Match(std::unique_ptr<TrainMC> player1,
             std::unique_ptr<TrainMC> player2, int32_t random_seed,
             std::unique_ptr<std::ofstream> log_file)
    : generator_(random_seed), players_{std::move(player1),
                                        std::move(player2)},
      log_file_(std::move(log_file)) {}

int32_t Match::to_play() const noexcept {
  return to_play_;
}

int32_t Match::num_requests() const noexcept {
  // Random players should finish their turn without evaluation
  assert(players_[to_play_] != nullptr);
  return players_[to_play_]->num_requests();
}

float Match::score() const noexcept {
  if (result_ == kResultLoss)
    return 0.0;
  if (result_ == kResultWin)
    return 1.0;
  return 0.5;
}

void Match::writeRequests(float *game_states) const noexcept {
  // Random players should finish their turn without evaluation
  assert(players_[to_play_] != nullptr);
  assert(game_states != nullptr);
  int32_t count = kGameStateSize * players_[to_play_]->num_requests();
  std::copy(to_eval_.get(), to_eval_.get() + count, game_states);
}

bool Match::doIteration(float eval[], float probs[]) {
  if (players_[to_play_] == nullptr) {
    return chooseMoveAndContinue();
  }
  // There can only be up to 1 random player
  bool done = players_[to_play_]->doIteration(eval, probs);
  // If we have completed a turn, we can choose a move
  if (done)
    return chooseMoveAndContinue();
  // Otherwise, the turn is not done so the game is not done
  return false;
}

void Match::writeEval(Node *node) const noexcept {
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

void Match::writeMoves() const noexcept {
  assert(log_file_ != nullptr);
  *log_file_ << "LEGAL MOVES:\n";
  // Print main line
  players_[to_play_]->root()->printMainLine(log_file_.get());
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
  Node *cur = players_[to_play_]->root()->first_child();
  int32_t edge_index = 0;
  while (cur != nullptr) {
    if (cur->child_id() == players_[to_play_]->root()->move_id(edge_index)) {
      moves.emplace_back(cur->visits(),
                         cur->evaluation() / static_cast<float>(cur->visits()),
                         players_[to_play_]->root()->probability(edge_index),
                         cur->child_id(), cur);
      cur = cur->next_sibling();
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
  for (size_t i = 1; i < moves.size(); ++i) {
    *log_file_ << Move{moves[i].move} << " V: " << moves[i].visits << " E: ";
    writeEval(moves[i].node);
    *log_file_ << " P: " << moves[i].probability << '\t';
  }
  *log_file_ << '\n';
}

void Match::writePreMoveLogs() const noexcept {
  assert(log_file_ != nullptr);
  *log_file_ << "TURN "
             << static_cast<int32_t>(players_[to_play_]->root()->depth())
             << "\nPLAYER " << static_cast<int32_t>(to_play_ + 1)
             << " TO PLAY\nVISITS: "
             << static_cast<int32_t>(players_[to_play_]->root()->visits())
             << '\n';
  *log_file_ << "POSITION EVALUATION: ";
  writeEval(players_[to_play_]->root());
  *log_file_ << '\n';
  writeMoves();
}

void Match::writeMoveChoice(int32_t choice) const noexcept {
  assert(log_file_ != nullptr);
  *log_file_ << "CHOSE MOVE " << Move{choice} << "\nNEW POSITION:\n"
             << root_->game() << "\n\n";
}

void Match::endGame() noexcept {
  assert(root_->terminal());
  // Set result
  if (root_->result() == kResultDraw) {
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
  players_[0]->null_root();
  players_[1]->null_root();
  to_eval_.reset();
  log_file_.reset();
}

int32_t Match::chooseMove() {
  // Random player
  if (players_[to_play_] == nullptr) {
    std::vector<int32_t> moves;
    for (int32_t i = 0; i < root_->num_legal_moves(); ++i) {
      moves.push_back(root_->move_id(i));
    }
    std::uniform_int_distribution<int32_t> dist(0, moves.size() - 1);
    return moves[dist(generator_)];
  }
  return players_[to_play_]->chooseMove();
}

bool Match::chooseMoveAndContinue() {
  bool need_eval = false;
  // Loop until we need an evaluation.
  // We could play many turns, for example when mating sequences are found by
  // both players
  while (!need_eval) {
    if (log_file_ != nullptr && players_[to_play_] != nullptr) {
      writePreMoveLogs();
    }
    int32_t choice = chooseMove();
    root_ = std::make_unique<Node>(root_->game(), nullptr, nullptr, choice,
                                   root_->depth() + 1);
    if (log_file_ != nullptr) {
      writeMoveChoice(choice);
    }
    // Check if the game is over
    if (players_[to_play_]->root()->terminal()) {
      endGame();
      return true;
    }
    // Go to next player
    to_play_ = 1 - to_play_;
    // Random player
    if (players_[to_play_] == nullptr) {
      continue;
    }
    // First time iterating the second player
    if (players_[to_play_]->uninitialized()) {
      players_[to_play_]->createRoot(players_[1 - to_play_]->root()->game(),
                                     players_[1 - to_play_]->root()->depth());
      // This is always false as the root requires an evaluation
      return players_[to_play_]->doIteration();
    }
    // It's possible that we need an evaluation for this
    // in the case that received move has not been searched
    need_eval = players_[to_play_]->receiveOpponentMove(
        choice, root_->get_game(), root_->depth());
    if (!need_eval) {
      // Otherwise, we search again.
      // If no evaluation is needed, this player also did all its iterations
      // without needing evaluations, so we loop again.
      // This can happen if a mating sequence is found
      need_eval = !players_[to_play_]->doIteration();
    }
  }
  return false;
}
