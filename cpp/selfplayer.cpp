#include "selfplayer.h"
#include "move.h"
#include "node.h"
#include "trainer.h"
#include "util.h"
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <utility>
#include <vector>
using std::cerr;
using std::make_pair;
using std::pair;

SelfPlayer::SelfPlayer(std::mt19937 *generator)
    : players{TrainMC{generator}, TrainMC{generator}}, to_play{0},
      generator{generator}, result{RESULT_NONE}, logging_file{nullptr} {
  players[0].game_states = &game_states;
  players[1].game_states = &game_states;
  samples.reserve(32);
}

SelfPlayer::SelfPlayer(std::mt19937 *generator, std::ofstream *logging_file)
    : players{TrainMC{generator}, TrainMC{generator}}, to_play{0},
      generator{generator}, result{RESULT_NONE}, logging_file{logging_file} {
  players[0].game_states = &game_states;
  players[1].game_states = &game_states;
  samples.reserve(32);
}

SelfPlayer::SelfPlayer(uintf seed, std::mt19937 *generator)
    : players{TrainMC{generator, true}, TrainMC{generator, true}}, to_play{0},
      generator{generator}, result{RESULT_NONE}, seed{seed}, logging_file{
                                                                 nullptr} {
  players[0].game_states = &game_states;
  players[1].game_states = &game_states;
}

SelfPlayer::SelfPlayer(uintf seed, std::mt19937 *generator,
                       std::ofstream *logging_file)
    : players{TrainMC{generator, true}, TrainMC{generator, true}}, to_play{0},
      generator{generator}, result{RESULT_NONE}, seed{seed}, logging_file{
                                                                 logging_file} {
  players[0].game_states = &game_states;
  players[1].game_states = &game_states;
}

SelfPlayer::~SelfPlayer() {
  delete logging_file;
  delete generator;
}

void SelfPlayer::do_first_iteration(float game_state[GAME_STATE_SIZE]) {
  players[0].do_first_iteration(game_state);
}

bool SelfPlayer::do_iteration(float evaluation[], float probabilities[],
                              float game_state[]) {
  bool done_turn =
      players[to_play].do_iteration(evaluation, probabilities, game_state);
  // If we have completed a turn
  // and we don't need to evaluate any positions
  // We can choose a move
  if (done_turn && players[to_play].eval_index == 0)
    return do_iteration(game_state);
  // Game is not complete
  return false;
}

bool SelfPlayer::do_iteration(float game_state[GAME_STATE_SIZE]) {

  bool need_evaluation = false;

  while (!need_evaluation) {

    if (logging_file != nullptr) {
      *logging_file << "TURN " << (uintf)players[to_play].root->depth << '\n'
                    << "PLAYER " << (uintf)(to_play + 1) << " TO PLAY\nVISITS: "
                    << (uintf)players[to_play].root->visits
                    << "\nPOSITION EVALUATION: " << std::fixed
                    << std::setprecision(6) << players[to_play].root->evaluation
                    << "\nLEGAL MOVES:\n";
      // Get and sort moves by visit count and evaluation
      std::vector<pair<pair<uintf, float>, pair<float, uintf>>> moves;
      Node *cur_child = players[to_play].root->first_child;
      uintf edge_index = 0;
      while (cur_child != nullptr) {
        if (cur_child->child_num ==
            players[to_play].root->edges[edge_index].move_id) {
          moves.emplace_back(
              make_pair(cur_child->visits, cur_child->evaluation),
              make_pair(players[to_play].root->get_probability(edge_index),
                        cur_child->child_num));
          cur_child = cur_child->next_sibling;
        }
        ++edge_index;
      }
      sort(moves.begin(), moves.end(),
           [](const pair<pair<uintf, float>, pair<float, uintf>> &A,
              const pair<pair<uintf, float>, pair<float, uintf>> &B) -> bool {
             if (A.first.first > B.first.first)
               return true;
             if (A.first.first < B.first.first)
               return false;
             if (A.first.second > B.first.second)
               return true;
             if (A.first.second < B.first.second)
               return false;
             return A.second.second < B.second.second;
           });
      // Print chosen move (should always be one with highest visit then eval
      // then lowest id)
      *logging_file << Move{moves[0].second.second}
                    << " V: " << moves[0].first.first << std::setprecision(2)
                    << " E: " << moves[0].first.second
                    << " P: " << moves[0].second.first << '\n';
      for (uintf i = 1; i < moves.size(); ++i) {
        *logging_file << Move{moves[i].second.second}
                      << " V: " << moves[i].first.first
                      << " E: " << moves[i].first.second
                      << " P: " << moves[i].second.first << '\t';
      }
      *logging_file << '\n';
    }

    // This function will automatically apply the move to the TrainMC
    // Also write samples
    std::array<float, GAME_STATE_SIZE> sample_state;
    std::array<float, NUM_MOVES> probability_sample;
    uintf move_choice = players[to_play].choose_move(sample_state.data(),
                                                     probability_sample.data());
    samples.emplace_back(sample_state, probability_sample);

    if (logging_file != nullptr) {
      *logging_file << "CHOSE MOVE " << Move{move_choice} << "\nNEW POSITION:\n"
                    << players[to_play].root->game << "\n\n";
    }

    // Check if the game is over
    if (players[to_play].root->is_terminal()) {
      // Get result
      result = players[to_play].root->result;
      // Log game result
      if (logging_file != nullptr) {
        if (result == RESULT_DRAW) {
          *logging_file << "GAME IS DRAWN.\n";
        } else {
          *logging_file << "PLAYER " << to_play + 1 << " WON!\n";
        }
      }
      delete players[0].root;
      players[0].root = nullptr;
      delete players[1].root;
      players[1].root = nullptr;
      // Return that the game is complete
      return true;
    }
    // Otherwise, iterate the other TrainMC
    else {
      to_play = 1 - to_play;
      // First time iterating the second TrainMC
      if (players[to_play].is_uninitialized()) {
        players[to_play].do_first_iteration(players[1 - to_play].root->game,
                                            players[1 - to_play].root->depth,
                                            game_state);
        // Game is not over, evaluation is needed
        // It cannot be a terminal state
        return false;
      } else {
        // It's possible that we need an evaluation for this
        // in the case that received move has not been searched
        need_evaluation = players[to_play].receive_opp_move(
            move_choice, game_state, players[1 - to_play].get_game(),
            players[1 - to_play].root->depth);
        if (!need_evaluation) {
          // Otherwise, we search again
          need_evaluation = !(players[to_play].search(game_state) &&
                              players[to_play].eval_index == 0);
          // If no evaluation is needed, this player also did all its iterations
          // without needing evaluations, so we loop again
          // This is unlikely, however
        }
      }
    }
  }

  // Game is not done
  return false;
}

uintf SelfPlayer::count_samples() const { return samples.size(); }

uintf SelfPlayer::write_samples(float *game_states, float *evaluation_samples,
                                float *probability_samples) const {
  uintf offset = 0;
  // The last player to play a move is the winner, except in a draw
  float evaluation = 1.0;
  if (result == RESULT_DRAW) {
    evaluation = 0.0;
  }
  // Start from back to front to figure out evaluations more easily
  for (intf i = samples.size() - 1; i >= 0; --i) {
    for (uintf j = 0; j < GAME_STATE_SIZE; ++j) {
      *(game_states + offset * GAME_STATE_SIZE * SYMMETRY_NUM + j) =
          samples[i].game_state[j];
    }
    *(evaluation_samples + offset * SYMMETRY_NUM) = evaluation;
    for (uintf j = 0; j < NUM_MOVES; ++j) {
      *(probability_samples + offset * NUM_MOVES * SYMMETRY_NUM + j) =
          samples[i].probabilities[j];
    }
    for (uintf k = 0; k < SYMMETRY_NUM - 1; ++k) {
      for (uintf j = 0; j < 4 * BOARD_SIZE; ++j) {
        *(game_states + offset * GAME_STATE_SIZE * SYMMETRY_NUM +
          (k + 1) * GAME_STATE_SIZE + j) =
            samples[i].game_state[space_symmetries[k][j / 4] * 4 + j % 4];
      }
      for (uintf j = 4 * BOARD_SIZE; j < GAME_STATE_SIZE; ++j) {
        *(game_states + offset * GAME_STATE_SIZE * SYMMETRY_NUM +
          (k + 1) * GAME_STATE_SIZE + j) = samples[i].game_state[j];
      }
      *(evaluation_samples + offset * SYMMETRY_NUM + (k + 1)) = evaluation;
      for (uintf j = 0; j < NUM_MOVES; ++j) {
        *(probability_samples + offset * NUM_MOVES * SYMMETRY_NUM +
          (k + 1) * NUM_MOVES + j) =
            samples[i].probabilities[move_symmetries[k][j]];
      }
    }
    ++offset;
    evaluation *= -1.0;
  }
  return samples.size();
}

float SelfPlayer::get_score() const {
  if (result == RESULT_WIN)
    return 1.0;
  if (result == RESULT_LOSS)
    return 0.0;
  return 0.5;
}

uintf SelfPlayer::count_nodes() const {
  return players[0].count_nodes() + players[1].count_nodes();
}
