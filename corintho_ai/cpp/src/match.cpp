#include "match.h"

#include <cassert>
#include <cstdint>

#include <fstream>
#include <iomanip>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "spdlog/sinks/basic_file_sink.h"
#include <spdlog/fmt/fmt.h>
#include <spdlog/spdlog.h>

#include "node.h"
#include "trainmc.h"

Match::Match(int32_t random_seed, Player player1, Player player2,
             const std::string &log_file)
    : generator_{std::mt19937(random_seed)},
      to_eval_{std::make_unique<float[]>(
          kGameStateSize *
          std::max(player1.max_searches, player2.max_searches))},
      players_{
          player1.random ? nullptr
                         : std::make_unique<TrainMC>(
                               &generator_, to_eval_.get(),
                               player1.max_searches, player1.searches_per_eval,
                               player1.c_puct, player1.epsilon, true),
          player2.random ? nullptr
                         : std::make_unique<TrainMC>(
                               &generator_, to_eval_.get(),
                               player2.max_searches, player2.searches_per_eval,
                               player2.c_puct, player2.epsilon, true)},
      ids_{player1.player_id, player2.player_id}, model_ids_{player1.model_id,
                                                             player2.model_id},
      logger_{log_file.empty() ? nullptr
                               : spdlog::basic_logger_mt("logger_" + log_file,
                                                         log_file, true)},
      debug_logger_{nullptr} {
  if (logger_)
    logger_->set_pattern("%C-%m-%d %H:%M:%S.%e %g:%!:%# %v");
}

int32_t Match::id(int32_t i) const noexcept {
  assert(i == 0 || i == 1);
  return ids_[i];
}

int32_t Match::to_play() const noexcept {
  return model_ids_[to_play_];
}

int32_t Match::num_requests() const noexcept {
  if (players_[to_play_] == nullptr)
    return 0;
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
  if (debug_logger_ != nullptr) {
    SPDLOG_LOGGER_INFO(debug_logger_, "HISTORY\n{}", history_);
    SPDLOG_LOGGER_INFO(
        debug_logger_,
        "TURN {}: PLAYER {} TO PLAY - VISITS: {} - POSITION EVALUATION: {}",
        static_cast<int32_t>(players_[to_play_]->root()->depth()),
        to_play_ + 1,
        static_cast<int32_t>(players_[to_play_]->root()->visits()),
        writeEval(players_[to_play_]->root()));
    SPDLOG_LOGGER_INFO(debug_logger_, "REQUESTS: {} - NODES: {} - DONE: {}",
                       players_[to_play_]->num_requests(),
                       players_[to_play_]->num_nodes(),
                       players_[to_play_]->done());
    SPDLOG_LOGGER_INFO(debug_logger_,
                       "LEGAL MOVES: {} - ALL VISITED: {} - KNOWN: {}",
                       players_[to_play_]->root()->num_legal_moves(),
                       players_[to_play_]->root()->all_visited(),
                       players_[to_play_]->root()->known());
    SPDLOG_LOGGER_INFO(debug_logger_, "POSITION:\n{}\n",
                       players_[to_play_]->root()->game().to_string());
    Node *cur = players_[to_play_]->root()->first_child();
    while (cur != nullptr) {
      SPDLOG_LOGGER_INFO(debug_logger_,
                         "MOVE: {}\tVISITS: {}\tEVAL: {}\nPOSITION:\n{}\n",
                         Move{cur->child_id()}.to_string(), cur->visits(),
                         writeEval(cur), cur->game().to_string());
      cur = cur->next_sibling();
    }
  }
  // There can only be up to 1 random player
  bool done = players_[to_play_]->doIteration(eval, probs);
  if (debug_logger_) SPDLOG_LOGGER_INFO(debug_logger_, "REQUESTS: {} - NODES: {} - DONE: {}",
                     players_[to_play_]->num_requests(),
                     players_[to_play_]->num_nodes(), done);
  // If we have completed a turn, we can choose a move
  if (done)
    return chooseMoveAndContinue();
  // Otherwise, the turn is not done so the game is not done
  return false;
}

std::string Match::writeEval(Node *node) const noexcept {
  assert(node != nullptr);
  assert(logger_ != nullptr);
  // There is a forced sequence
  if (node->result() != kResultNone) {
    return strResult(node->result());
  }
  // Log the evaluation
  return fmt::format("{:.3f}", node->evaluation() / node->visits());
}

void Match::writeMoves() const noexcept {
  assert(logger_ != nullptr);
  SPDLOG_LOGGER_INFO(logger_, "LEGAL MOVES:");
  if (debug_logger_ != nullptr) {
    SPDLOG_LOGGER_INFO(debug_logger_, "LEGAL MOVES:");
  }
  // Print main line
  players_[to_play_]->root()->printMainLine(logger_);
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
  std::string move_string = "";
  // The first move is already printed in the main line
  for (size_t i = 1; i < moves.size(); ++i) {
    move_string += fmt::format(
        "{} V: {} E: {} P: {:.3f}\t", Move{moves[i].move}.to_string(),
        moves[i].visits, writeEval(moves[i].node), moves[i].probability);
  }
  SPDLOG_LOGGER_INFO(logger_, move_string);
  if (debug_logger_ != nullptr) {
    SPDLOG_LOGGER_INFO(debug_logger_, move_string);
  }
}

void Match::writePreMoveLogs() const noexcept {
  assert(logger_ != nullptr);
  SPDLOG_LOGGER_INFO(
      logger_, "TURN {}",
      static_cast<int32_t>(players_[to_play_]->root()->depth()));
  SPDLOG_LOGGER_INFO(logger_, "PLAYER {} TO PLAY", to_play_ + 1);
  SPDLOG_LOGGER_INFO(
      logger_, "VISITS: {}",
      static_cast<int32_t>(players_[to_play_]->root()->visits()));
  SPDLOG_LOGGER_INFO(logger_, "POSITION EVALUATION: {}",
                     writeEval(players_[to_play_]->root()));
  writeMoves();
}

void Match::writeMoveChoice(int32_t choice) noexcept {
  assert(logger_ != nullptr);
  SPDLOG_LOGGER_INFO(logger_, "CHOSE MOVE {}", Move{choice}.to_string());
  SPDLOG_LOGGER_INFO(logger_, "NEW POSITION:\n{}\n",
                     players_[to_play_]->root()->game().to_string());
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
  if (logger_ != nullptr) {
    if (result_ == kResultDraw) {
      SPDLOG_LOGGER_INFO(logger_, "GAME IS DRAWN");
    } else {
      SPDLOG_LOGGER_INFO(logger_, "PLAYER {} WON!", to_play_ + 1);
    }
  }
  // Delete the players
  // We cannot delete the Match yet as it contains training samples
  // and results which will be collected at the end
  if (players_[0] != nullptr) {
    players_[0]->null_root();
  }
  if (players_[1] != nullptr) {
    players_[1]->null_root();
  }
  to_eval_.reset();
  logger_.reset();
}

int32_t Match::chooseMove() {
  // Random player
  if (players_[to_play_] == nullptr) {
    std::vector<int32_t> moves;
    for (int32_t i = 0; i < root_->num_legal_moves(); ++i) {
      moves.push_back(root_->move_id(i));
    }
    std::uniform_int_distribution<int32_t> dist(0, moves.size() - 1);
    int32_t choice = moves[dist(generator_)];
    return choice;
  }
  return players_[to_play_]->chooseMove();
}

bool Match::chooseMoveAndContinue() {
  bool need_eval = false;
  // Loop until we need an evaluation.
  // We could play many turns, for example when mating sequences are found by
  // both players
  while (!need_eval) {
    if (logger_ != nullptr && players_[to_play_] != nullptr) {
      writePreMoveLogs();
    }
    int32_t choice = chooseMove();
    if (debug_logger_) SPDLOG_LOGGER_INFO(debug_logger_, "CHOSE MOVE {}", Move{choice}.to_string());
    root_ = std::make_unique<Node>(root_->game(), nullptr, nullptr, choice,
                                   root_->depth() + 1);
    if (logger_ != nullptr) {
      writeMoveChoice(choice);
    }
    history_ += Move{choice}.to_string() + "\n" +
              players_[to_play_]->root()->game().to_string() + "\n";
    // Check if the game is over
    if (root_->terminal()) {
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
      players_[to_play_]->createRoot(root_->game(), root_->depth());
      // This is always false as the root requires an evaluation
      players_[to_play_]->doIteration();
      return false;
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

void Match::addDetailedLog(const std::string &filename) {
  debug_logger_ =
      spdlog::basic_logger_mt("debug_logger_" + filename, filename, true);
  debug_logger_->set_pattern("%C-%m-%d %H:%M:%S.%e %g:%!:%# %v");
  players_[0]->addDetailedLog(debug_logger_);
  players_[1]->addDetailedLog(debug_logger_);
}