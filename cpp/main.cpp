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

    cerr << "sizeof(uintf): " << sizeof(uintf) << '\n';
    cerr << "sizeof(Move): " << sizeof(Move) << '\n';
    cerr << "sizeof(Game): " << sizeof(Game) << '\n';
    cerr << "sizeof(Node): " << sizeof(Node) << '\n';
    cerr << "sizeof(TrainMC): " << sizeof(TrainMC) << '\n';
    cerr << "sizeof(SelfPlayer): " << sizeof(SelfPlayer) << '\n';
    cerr << "sizeof(ofstream): " << sizeof(ofstream) << '\n';
    cerr << "sizeof(Trainer): " << sizeof(Trainer) << '\n';
    cerr << "sizeof(mt19937): " << sizeof(mt19937) << '\n';

}

void print_game() {

    cerr << Game() << '\n';

}

void print_line_breakers(uintf line) {

    for (uintf i = 0; i < NUM_TOTAL_MOVES; ++i) {
        if (line_breakers[line][i]) {
            cerr << Move{i} << ' ';
        }
    }
    cerr << '\n';
}

// Basic run. 
void test_basic_run() {

    uintf num_games = 1;

    auto trainer = Trainer{num_games, 1, 100, 1.0, 0.25, "logging"};

    float evaluations[num_games], probabilities[num_games][NUM_TOTAL_MOVES],
    dirichlet_noise[num_games][NUM_MOVES], game_states[num_games][GAME_STATE_SIZE];

    mt19937 generator(0);
    uniform_real_distribution<float> random_evals(-1.0, 1.0), random_probabilities(0.0, 1.0), random_noise(-0.1, 0.1);

    auto start = chrono::high_resolution_clock::now();
    uintf counter = 0;
    while (true) {
        //trainer.rehash_if_full();
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
        bool is_done = trainer.do_iteration(evaluations, probabilities, dirichlet_noise, game_states);
        if (is_done) break;
        ++counter;
        if (counter % 1000 == 0) cerr << "Complete iteration " << counter << '\n';
    }
    auto stop = chrono::high_resolution_clock::now();

    auto duration = duration_cast<chrono::seconds>(stop - start);

    cerr << "Took " << duration.count() << " seconds!\n";
 
// To get the value of duration use the count()
// member function on the duration object
cerr << duration.count() << endl;

}

int main() {

    print_class_sizes();

    print_game();

    print_line_breakers(0);

    test_basic_run();

}
