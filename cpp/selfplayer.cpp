#include "selfplayer.h"

SelfPlayer::SelfPlayer(Trainer *trainer): testing{false}, logging{false},
players{TrainMC{trainer}, TrainMC{trainer}}, to_play{0}, trainer{trainer} {}

SelfPlayer::SelfPlayer(bool, Trainer *trainer): testing{false}, logging{true},
players{TrainMC{true, trainer}, TrainMC{true, trainer}}, to_play{0}, trainer{trainer} {}

SelfPlayer::SelfPlayer(uintf seed, Trainer *trainer): testing{true}, logging{false}, seed{seed},
players{TrainMC{true, logging, trainer}, TrainMC{true, logging, trainer}}, to_play{0}, trainer{trainer} {}

SelfPlayer::SelfPlayer(bool, uintf seed, Trainer *trainer): testing{true}, logging{true}, seed{seed},
players{TrainMC{true, logging, trainer}, TrainMC{true, logging, trainer}}, to_play{0}, trainer{trainer} {}

// It is relatively costless to detect when a SelfPlayer will be called by Trainer for the first time
// Instead of having an extra if statement in do_iteration, we can split it into 2 functions
// game_state is where the game_state should be written to be evaluated
void SelfPlayer::do_first_iteration(float game_state[GAME_STATE_SIZE]) {
    players[0].do_first_iteration(game_state);
}

// Can we store the game_state array in SelfPlayer instead getting it passed each time?
// Depends on if we can keep the same one each time
void SelfPlayer::do_iteration(float evaluation_result, float probability_result[NUM_TOTAL_MOVES],
float dirichlet_noise[NUM_MOVES], float game_state[GAME_STATE_SIZE]) {

    bool need_evaluation = players[to_play].do_iteration(evaluation_result, probability_result,
    dirichlet_noise, game_state);
    // Done iterations
    while (!need_evaluation) {
        uintf move_choice = players[to_play].choose_move();
        // We can check if the game is over
        if (trainer->get_node(players[to_play].root)->is_terminal()) {
            trainer->delete_tree(players[to_play].root);
            // Write training samples using game result, make all nodes stale
            return;
        }
        else {
            to_play = 1 - to_play;
            // First time iterating the second TrainMC
            if (players[to_play].cur_node == nullptr) {
                players[to_play].do_first_iteration(players[1 - to_play].game, game_state);
                need_evaluation = true
            }
            else {
                // it's possible that we need an evaluation for this
                need_evaluation = players[to_play].receive_opp_move(move_choice, game_state);
                if (!need_evaluation) {
                    need_evaluation = players[to_play].do_iteration(game_state);
                }
            }
        }
    }
}