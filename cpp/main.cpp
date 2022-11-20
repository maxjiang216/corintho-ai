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

    int num_games = 100;

    auto trainer = Trainer(false, num_games, 0, 1600, 1.0, 0.25);

    float evaluations[num_games], probabilities[num_games][NUM_TOTAL_MOVES],
    dirichlet_noise[num_games][NUM_MOVES], game_states[num_games][GAME_STATE_SIZE];

    trainer.do_iteration(evaluations, probabilities, dirichlet_noise, game_states);

}

int main() {

    print_class_sizes();

    test_basic_run();

}
