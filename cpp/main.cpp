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

    uintf num_games = 300;

    auto trainer = Trainer{num_games, 0, 200, 1.0, 0.25};

    float evaluations[num_games], probabilities[num_games][NUM_TOTAL_MOVES],
    dirichlet_noise[num_games][NUM_MOVES], game_states[num_games][GAME_STATE_SIZE];

    mt19937 generator(0);
    uniform_real_distribution<float> random_evals(-1.0, 1.0), random_probabilities(0.0, 1.0), random_noise(-0.1, 0.1);

    auto start = chrono::high_resolution_clock::now();
    uintf counter = 0;
    while (true) {
        trainer.rehash_if_full();
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
        cout << "Complete iteration " << ++counter << '\n';
    }
    auto stop = chrono::high_resolution_clock::now();

    auto duration = duration_cast<chrono::seconds>(stop - start);

    cout << "Took " << duration.count() << " seconds!\n";
 
// To get the value of duration use the count()
// member function on the duration object
cout << duration.count() << endl;

}

int main() {

    print_class_sizes();

    test_basic_run();

}
