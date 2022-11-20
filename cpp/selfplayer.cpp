#include "selfplayer.h"

SelfPlayer::SelfPlayer(Trainer *trainer): testing{false}, logging{false},
players{TrainMC{trainer}, TrainMC{trainer}}, to_play{0}, trainer{trainer} {}

SelfPlayer::SelfPlayer(bool, Trainer *trainer): testing{false}, logging{true},
players{TrainMC{true, trainer}, TrainMC{true, trainer}}, to_play{0}, trainer{trainer} {}

SelfPlayer::SelfPlayer(uint seed, Trainer *trainer): testing{true}, logging{false}, seed{seed},
players{TrainMC{true, logging, trainer}, TrainMC{true, logging, trainer}}, to_play{0}, trainer{trainer} {}

SelfPlayer::SelfPlayer(bool, uint seed, Trainer *trainer): testing{true}, logging{true}, seed{seed},
players{TrainMC{true, logging, trainer}, TrainMC{true, logging, trainer}}, to_play{0}, trainer{trainer} {}

// It is relatively costless to detect when a SelfPlayer will be called by Trainer for the first time
// Instead of having an extra if statement in do_iteration, we can split it into 2 functions
// game_state is where the game_state should be written to be evaluated
void SelfPlayer::do_first_iteration(float game_state[]) {
    players[0].do_first_iteration(game_state);
}

void SelfPlayer::do_iterations(float evaluation_result, float probability_result[NUM_TOTAL_MOVES],
float dirichlet_noise[NUM_MOVES], float game_state[]) {
    bool need_evaluation = players[to_play].do_iterations(evaluation_result, probability_result, dirichlet_noise, game_state);
    // Done iterations
    while (!need_evaluation) {
        uint move_choice = players[to_play].choose_move();
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
                players[to_play].receive_opp_move(move_choice);
                need_evaluation = players[to_play].do_iterations(evaluation_result, probability_result, dirichlet_noise, game_state);
            }
        }
    }
}