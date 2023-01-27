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
    cerr << '"' << Move{i} << "\": " << i << ",\n";
  }
}

void print_game() { cerr << Game() << '\n'; }

void print_line_breakers(uintf line) {
  print_line(line);
  cerr << '\n';
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

// Basic run.
void test_basic_run() {

  uintf num_games = 600, num_iterations = 1600, searches_per_eval = 8;

  auto trainer =
      Trainer{num_games,         10,        num_iterations, 1.0, 0.25, 1,
              searches_per_eval, "logging", 2003,           true};

  float evaluations[num_games * searches_per_eval],
      probabilities[num_games * NUM_MOVES * searches_per_eval],
      game_states[num_games * GAME_STATE_SIZE * searches_per_eval];

  mt19937 generator(0);
  uniform_real_distribution<float> random_evals(-1.0, 1.0),
      random_probabilities(0.0, 1.0);

  cerr << "Ready to play!\n";

  uintf to_play = 0;

  auto start = chrono::high_resolution_clock::now();

  trainer.do_iteration(evaluations, probabilities, to_play);
  uintf counter = 1;
  while (true) {
    uintf num_requests = trainer.write_requests(game_states, to_play);
    if (num_requests > 0) {
      for (uintf i = 0; i < num_requests; ++i) {
        if (to_play == 0) {
          evaluations[i] = random_evals(generator);
        } else {
          evaluations[i] = 0.0;
        }
        float sum = 0.0;
        for (uintf j = 0; j < NUM_MOVES; ++j) {
          probabilities[i * NUM_MOVES + j] = random_probabilities(generator);
          sum += probabilities[i * NUM_MOVES + j];
        }
        for (uintf j = 0; j < NUM_MOVES; ++j) {
          probabilities[i * NUM_MOVES + j] /= sum;
        }
      }
      bool is_done = trainer.do_iteration(evaluations, probabilities, to_play);
      if (is_done)
        break;
      ++counter;
      if (counter % 10 == 0) {
        cerr << "Complete evaluation " << counter << '\n';
        /*uintf num_nodes = trainer.count_nodes();
        cerr << "Nodes per game per iteration: "
             << (float)num_nodes / (float)num_games / (float)num_iterations
             << '\n';*/
      }
    } else {
      to_play = 1 - to_play;
    }
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

  print_moves();

  print_game();

  for (int i = 0; i < 102; ++i) {
    cerr << i << '\n';
    print_line_breakers(i);
  }

  test_basic_run();
}