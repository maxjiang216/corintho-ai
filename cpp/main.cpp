#include "game.h"
#include "move.h"
#include "node.h"
#include "selfplayer.h"
#include "trainer.h"
#include "trainmc.h"
#include "util.h"
#include <bits/stdc++.h>

using namespace std;

struct Edge {
    uint16s move_id : 7, probability : 9;
  };

// Print the size of the classes we're using
void print_class_sizes() {

  cerr << "sizeof(uintf): " << sizeof(uintf) << '\n';
  cerr << "sizeof(Move): " << sizeof(Move) << '\n';
  cerr << "sizeof(Game): " << sizeof(Game) << '\n';
  cerr << "sizeof(Node::Edge): " << sizeof(Edge) << '\n';
  cerr << "sizeof(Node): " << sizeof(Node) << '\n';
  cerr << "sizeof(TrainMC): " << sizeof(TrainMC) << '\n';
  cerr << "sizeof(SelfPlayer): " << sizeof(SelfPlayer) << '\n';
  cerr << "sizeof(ofstream): " << sizeof(ofstream) << '\n';
  cerr << "sizeof(Trainer): " << sizeof(Trainer) << '\n';
  cerr << "sizeof(mt19937): " << sizeof(mt19937) << '\n';
}

void print_moves() {
  for (uintf i = 0; i < NUM_MOVES; ++i) {
    cerr << i << ' ' << Move{i} << '\n';
  }
}

void print_game() { cerr << Game() << '\n'; }

void print_line_breakers(uintf line) {

  for (uintf i = 0; i < NUM_MOVES; ++i) {
    if (line_breakers[line][i]) {
      cerr << 1;
    } else {
      cerr << 0;
    }
  }
  cerr << '\n';
  for (uintf i = 0; i < NUM_MOVES; ++i) {
    if (line_breakers[line][i]) {
      cerr << Move{i} << ' ';
    }
  }
  cerr << '\n';
}
/*
// Basic run.
void test_basic_run() {

  uintf num_games = 3000;

  auto trainer = Trainer{num_games, 1, 200, 1.0, 0.25, "logging", 2003};

  float evaluations[num_games], probabilities[num_games * NUM_TOTAL_MOVES],
      game_states[num_games * GAME_STATE_SIZE];

  mt19937 generator(0);
  uniform_real_distribution<float> random_evals(-1.0, 1.0),
      random_probabilities(0.0, 1.0);

  auto start = chrono::high_resolution_clock::now();
  uintf counter = 0;
  while (true) {
    for (uintf i = 0; i < num_games; ++i) {
      evaluations[i] = random_evals(generator);
      float sum = 0.0;
      for (uintf j = 0; j < NUM_TOTAL_MOVES; ++j) {
        probabilities[i * NUM_TOTAL_MOVES + j] =
            random_probabilities(generator);
        sum += probabilities[i * NUM_TOTAL_MOVES + j];
      }
      for (uintf j = 0; j < NUM_TOTAL_MOVES; ++j) {
        probabilities[i * NUM_TOTAL_MOVES + j] /= sum;
      }
    }
    bool is_done =
        trainer.do_iteration(evaluations, probabilities, game_states);
    if (is_done)
      break;
    ++counter;
    if (counter % 1000 == 0) {
      cerr << "Complete iteration " << counter << '\n';
      trainer.rehash_if_full();
    }
  }
  auto stop = chrono::high_resolution_clock::now();

  auto duration = duration_cast<chrono::seconds>(stop - start);

  cerr << "Took " << duration.count() << " seconds!\n";

  // To get the value of duration use the count()
  // member function on the duration object
  cerr << duration.count() << endl;
}
*/

int main() {

  print_class_sizes();

  print_moves();

  print_game();

  for (int i = 0; i < 102; ++i) {
    cerr << i << '\n';
    print_line_breakers(i);
  }

  // test_basic_run();
}