#ifndef MANAGER_H
#define MANAGER_H
#include "trainer.h"
#include <vector>

using std::vector;

class Manager {

  uintf games_each;
  vector<Trainer *> trainers;
  vector<bool> is_done;

public:
  Manager() = default;
  Manager(uintf num_games, uintf num_logged, uintf num_iterations, float c_puct,
          float epsilon, const std::string &logging_folder, uintf random_seed,
          uintf processes);
  ~Manager();

  bool do_iteration(float evaluations[], float probabilities[],
                    float dirichlet[], float game_states[]);

  uintf count_samples() const;
  void write_samples(float *game_states, float *evaluation_samples,
                     float *probability_samples) const;
};

#endif