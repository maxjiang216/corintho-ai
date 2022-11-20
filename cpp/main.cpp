#include <bits/stdc++.h>
#include "move.h"
#include "game.h"
#include "node.h"
#include "trainmc.h"
#include "selfplayer.h"
#include "trainer.h"
#include "util.h"

using namespace std;

// Print the size of the classes we're using
void print_class_sizes() {

    cout << "sizeof(uintf): " << sizeof(uintf) << '\n';
    cout << "sizeof(Move): " << sizeof(Move) << '\n';
    cout << "sizeof(Game): " << sizeof(Game) << '\n';
    cout << "sizeof(Node): " << sizeof(Node) << '\n';
    cout << "sizeof(TrainMC): " << sizeof(TrainMC) << '\n';
    cout << "sizeof(SelfPlayer): " << sizeof(SelfPlayer) << '\n';
    cout << "sizeof(Trainer): " << sizeof(Trainer) << '\n';
    cout << "sizeof(mt19937): " << sizeof(mt19937) << '\n';
}

// Basic run. 
void test_basic_run() {

    for (uintf i = 0; j < NUM_MOVES; )
    for (uintf i = 0; i < 102; ++i) {
        for (uintf j = 0; j < NUM_MOVES; ++j) {
            cout << line_breakers[i][j];
        }
        cout << '\n';
    }

    uintf num_games = 1;

    auto trainer = Trainer(false, num_games, 0, 1600, 1.0, 0.25);

    float evaluations[num_games], probabilities[num_games][NUM_TOTAL_MOVES],
    dirichlet_noise[num_games][NUM_MOVES], game_states[num_games][GAME_STATE_SIZE];

    mt19937 generator(0);
    uniform_real_distribution<float> random_evals(-1.0, 1.0), random_probabilities(0.0, 1.0), random_noise(-0.1, 0.1);

    for (uintf i = 0; i < 100; ++i) {
        for (uintf i = 0; i < num_games; ++i) {
            evaluations[i] = random_evals(generator);
            float sum = 0.0;
            for (uintf j = 0; j < NUM_TOTAL_MOVES; ++j) {
                probabilities[i][j] = random_probabilities(generator);
                sum += probabilities[i][j];
            }
            for (uintf j = 0; j < NUM_TOTAL_MOVES; ++j) {
                probabilities[i][j] /= sum;
            }
            for (uintf j = 0; j < NUM_MOVES; ++j) {
                dirichlet_noise[i][j] = random_noise(generator);
            }
        }
        trainer.do_iteration(evaluations, probabilities, dirichlet_noise, game_states);
        cout << "Complete iteration " << i + 1 << '\n';
    }

}

int main() {

    print_class_sizes();

    test_basic_run();

}
