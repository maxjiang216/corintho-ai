#include "selfplayer.h"
#include "trainer.h"
#include "move.h"
#include "util.h"
#include <fstream>
#include <string>
#include <iostream>
using std::cerr;

SelfPlayer::SelfPlayer(Trainer *trainer): players{TrainMC{trainer}, TrainMC{trainer}}, to_play{0},
                                          logging{false}, logging_file{nullptr}, trainer{trainer} {}

SelfPlayer::SelfPlayer(Trainer *trainer, uintf id, const std::string &logging_folder):
                       players{TrainMC{trainer, true}, TrainMC{trainer, true}},
                       to_play{0}, logging{true},
                       logging_file{new std::ofstream{logging_folder + "/game_" +
                                                                   std::to_string(id) + ".txt",
                                                  std::ofstream::out}},
                       trainer{trainer} {}

SelfPlayer::SelfPlayer(uintf seed, Trainer *trainer):
                       players{TrainMC{false, trainer, true}, TrainMC{false, trainer, true}},
                       to_play{0}, logging{false}, logging_file{nullptr}, trainer{trainer} {}

SelfPlayer::SelfPlayer(uintf seed, Trainer *trainer, uintf id, const std::string &logging_folder):
                       players{TrainMC{true, trainer, true}, TrainMC{true, trainer, true}},
                       to_play{0}, logging{true},
                       logging_file{new std::ofstream{logging_folder + "/game_" +
                                                                   std::to_string(id) + ".txt",
                                                  std::ofstream::out}},
                       trainer{trainer} {}

SelfPlayer::~SelfPlayer() {
    if (logging_file != nullptr) {
        delete logging_file;
    }
}

void SelfPlayer::do_first_iteration(float game_state[GAME_STATE_SIZE]) {
    cerr << "SelfPlayer::do_first_iteration!\n";
    players[0].do_first_iteration(game_state);
}

bool SelfPlayer::do_iteration(float evaluation, float probabilities[NUM_TOTAL_MOVES],
                              float dirichlet_noise[NUM_MOVES], float game_state[GAME_STATE_SIZE]) {
    bool need_evaluation = players[to_play].do_iteration(evaluation, probabilities,
                                                         dirichlet_noise, game_state);
    // If we don't need an evaluation
    // Then we have completed a turn
    if (!need_evaluation) return do_iteration(game_state);
    return false;
}

bool SelfPlayer::do_iteration(float evaluation_1, float probabilities_1[NUM_TOTAL_MOVES],
                              float evaluation_2, float probabilities_2[NUM_TOTAL_MOVES],
                              float dirichlet_noise[NUM_MOVES], float game_state[GAME_STATE_SIZE]) {
    bool need_evaluation;
    if (to_play == seed) {
        need_evaluation = players[to_play].do_iteration(evaluation_1, probabilities_1,
                                                        dirichlet_noise, game_state);
    }
    else {
        need_evaluation = players[to_play].do_iteration(evaluation_2, probabilities_2,
                                                        dirichlet_noise, game_state);
    }
    // If we don't need an evaluation
    // Then we have completed a turn
    if (!need_evaluation) return do_iteration(game_state);
    return false;
}

uintf SelfPlayer::get_root(uintf player_num) const {
    return players[player_num].get_root();
}
    
bool SelfPlayer::do_iteration(float game_state[GAME_STATE_SIZE]) {

    bool need_evaluation = false;

    while (!need_evaluation) {

        // This function will automatically apply the move to the TrainMC
        uintf move_choice = players[to_play].choose_move();

        if (logging) {
            *logging_file << trainer->get_node(players[to_play].get_root())->get_game() << '\n';
        }

        // Check if the game is over
        if (trainer->get_node(players[to_play].get_root())->is_terminal()) {
            // Make all nodes stale
            trainer->delete_tree(players[0].get_root());
            trainer->delete_tree(players[1].get_root());
            // Write training samples using game result
            // Exit the loop, nothing more to be done
            // Return that the game is complete
            return true;
        }
        // Otherwise, iterate the other TrainMC
        else {
            to_play = 1 - to_play;
            // First time iterating the second TrainMC
            if (players[to_play].is_uninitialized()) {
                players[to_play].do_first_iteration(players[1 - to_play].get_game(), game_state);
                need_evaluation = true;
            }
            else {
                // It's possible that we need an evaluation for this
                // in the case that received move has not been searched
                need_evaluation = players[to_play].receive_opp_move(move_choice, game_state,
                                                                    players[1 - to_play].get_game(),
                                                                    players[1 - to_play].get_depth());
                if (!need_evaluation) {
                    // If no evaluation is need, this player also did all its iterations
                    // without needing evaluations, so we loop again
                    need_evaluation = players[to_play].do_iteration(game_state);
                }
            }
        }

    }

    // Game is not done
    return false;

}