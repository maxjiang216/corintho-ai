#include "trainer.h"

// Determine best starting size empirically (per game)
const int HASH_TABLE_SIZE = 2048;

Trainer::Trainer(int num_games, int num_logged, int num_iterations, float states_to_evaluate[][GAME_STATE_SIZE], float c_puct, float epsilon):
                 num_games{num_games}, num_logged{num_logged}, num_iterations{num_iterations}, states_to_evaluate{states_to_evaluate} {
    hash_table.reserve(HASH_TABLE_SIZE * num_games);
    games.reserve(num_games);
    for (int i = 0; i < num_games; ++i) {
        games.push_back(SelfPlayer());
    }
}

void do_iteration(float evaluation_results[], float probability_results[][NUM_LEGAL_MOVES]) {
    for (int i = 0; i < num_games; ++i) {
        // Pass neural net results
        if (i / num_iterations < iterations_done) {
            games[i].do_iteration(evaluation_results[i], probability_results[i]);
        }
        // First iteration
        else if (i / num_iterations == iterations_done) {
            games[i].do_first_iteration();
        }
    }
    ++iterations_done;
}