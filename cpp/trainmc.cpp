#include "trainmc.h"
#include "move.h"
#include "trainer.h"
#include <array>
#include <bitset>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
using std::cerr;

using std::bitset;

TrainMC::TrainMC(Trainer *trainer)
    : cur_node{nullptr},
      iterations_done{1}, testing{false}, logging{false}, trainer{trainer} {}

TrainMC::TrainMC(Trainer *trainer, bool)
    : cur_node{nullptr},
      iterations_done{1}, testing{false}, logging{true}, trainer{trainer} {}

TrainMC::TrainMC(bool logging, Trainer *trainer, bool)
    : cur_node{nullptr},
      iterations_done{1}, testing{true}, logging{true}, trainer{trainer} {}

void TrainMC::do_first_iteration(float game_state[GAME_STATE_SIZE]) {

  // Create the root node
  root = trainer->place_root();

  cur = root;
  cur_node = trainer->get_node(root);

  cur_node->write_game_state(game_state);
}

void TrainMC::do_first_iteration(const Game &game,
                                 float game_state[GAME_STATE_SIZE]) {

  // Create the new root node
  root = trainer->place_root(game, 1);
  cur = root;
  cur_node = trainer->get_node(root);

  cur_node->write_game_state(game_state);
}

bool TrainMC::do_iteration(float game_state[GAME_STATE_SIZE]) {
  return search(game_state);
}

bool TrainMC::do_iteration(float evaluation,
                           const float probabilities[NUM_TOTAL_MOVES],
                           float game_state[GAME_STATE_SIZE]) {
  receive_evaluation(evaluation, probabilities);
  return search(game_state);
}

uintf TrainMC::choose_move(
    std::array<float, GAME_STATE_SIZE> &game_state,
    std::array<float, NUM_TOTAL_MOVES> &probability_sample) {

  // Before moving down, read the samples from the root node
  Node *root_node = trainer->get_node(root);
  root_node->write_game_state(game_state);
  for (uintf i = 0; i < NUM_TOTAL_MOVES; ++i) {
    if (root_node->has_visited(i)) {
      probability_sample[i] =
          (float)trainer->get_node(trainer->find_next(root_node->get_seed(), i))
              ->get_visits() /
          ((float)root_node->get_visits() - 1.0);
    } else {
      probability_sample[i] = 0.0;
    }
  }

  uintf move_choice = 0;

  // In training, choose weighted random
  // For the first few moves
  if (root_node->get_depth() < NUM_OPENING_MOVES && !testing) {
    uintf total = 0, children_visits[NUM_MOVES];
    for (uintf i = 0; i < NUM_MOVES; ++i) {
      if (root_node->has_visited(i)) {
        children_visits[i] =
            trainer->get_node(trainer->find_next(root_node->get_seed(), i))
                ->get_visits();
        total += children_visits[i];
      }
    }
    uintf cur_total = 0, target = trainer->generate() % total;
    for (uintf i = 0; i < NUM_MOVES; ++i) {
      if (root_node->has_visited(i)) {
        cur_total += children_visits[i];
        if (cur_total >= target) {
          move_choice = i;
          break;
        }
      }
    }
  }

  else {

    // Otherwise, choose randomly between the moves with the most
    // visits/searches. Random offset is the easiest way to randomly break ties
    uintf id = trainer->generate() % NUM_MOVES, max_visits = 0;
    for (uintf i = 0; i < NUM_MOVES; ++i) {
      uintf cur_move = (id + i) % NUM_MOVES;
      if (root_node->has_visited(cur_move)) {
        uintf cur_visits =
            trainer
                ->get_node(trainer->find_next(root_node->get_seed(), cur_move))
                ->get_visits();
        if (cur_visits > max_visits) {
          max_visits = cur_visits;
          move_choice = cur_move;
        }
      }
    }
  }

  uintf old_root = root;
  root = trainer->find_next(root_node->get_seed(), move_choice);
  trainer->move_down(old_root, move_choice);
  // Set cur_node here, but this is not needed if the game is complete
  // We can try to find a better place, but it's pretty insignificant and avoid
  // possible hassle
  cur = root;
  cur_node = trainer->get_node(cur);
  // Do this to avoid clashing parents
  // All nodes with possibly overwritten parents
  // Should either be stale
  // Our have itself as its parent
  cur_node->null_parent();
  iterations_done = 0;

  return move_choice;
}

bool TrainMC::receive_opp_move(uintf move_choice,
                               float game_state[GAME_STATE_SIZE],
                               const Game &game, uintf depth) {

  // If we have visited the node before
  // Simply move the tree down
  if (trainer->get_node(root)->has_visited(move_choice)) {
    uintf old_root = root;
    root = trainer->find_next(trainer->get_node(root)->get_seed(), move_choice);
    // We should stale things afterwards now
    // Since find_next skips over stale nodes
    trainer->move_down(old_root, move_choice);
    cur = root;
    cur_node = trainer->get_node(cur);
    iterations_done = 0;
    // we don't need an evaluation
    return false;
  }

  // The node doesn't exist
  // Practically (with sufficient searches) this probably won't happen but it is
  // theoretically possible Maybe we could empirically track if it happens
  else {
    trainer->delete_tree(root);
    // Copy opponent game state into our root
    root = trainer->place_root(game, depth);
    cur = root;
    cur_node = trainer->get_node(root);
    // We need an evaluation
    cur_node->write_game_state(game_state);
    // this is the first iteration of the turn
    iterations_done = 1;
    // we need an evaluation
    return true;
  }
}

uintf TrainMC::get_root() const { return root; }

const Game &TrainMC::get_game() const { return cur_node->get_game(); }

uintf TrainMC::get_depth() const { return cur_node->get_depth(); }

bool TrainMC::is_uninitialized() const { return cur_node == nullptr; }

void TrainMC::set_statics(uintf new_max_iterations, float new_c_puct,
                          float new_epsilon) {
  max_iterations = new_max_iterations;
  c_puct = new_c_puct;
  epsilon = new_epsilon;
}

void TrainMC::receive_evaluation(float evaluation,
                                 const float probabilities[NUM_TOTAL_MOVES]) {

  // Apply the legal move filter
  // and keep track of the total sum so we can normalize afterwards
  float sum = 0.0, p[NUM_TOTAL_MOVES];
  uintf legal_move_num = 0;
  for (uintf i = 0; i < NUM_MOVES; ++i) {
    if (cur_node->is_legal(i)) {
      p[i] = probabilities[i];
      sum += p[i];
      ++legal_move_num;
    } else {
      p[i] = 0.0;
    }
  }

  // Multiplying by this is more efficient
  float scalar = 1.0 / sum * (1 - epsilon);

  // Generate dirichlet noise (approximation)
  sum = 0.0;
  float *dirichlet_noise = new float[legal_move_num];

  for (uintf i = 0; i < legal_move_num; ++i) {
    dirichlet_noise[i] = gamma_samples[trainer->generate() % GAMMA_BUCKETS];
    sum += dirichlet_noise[i];
  }
  float dirichlet_scalar = 1.0 / sum * epsilon;

  // Normalize probabilities and apply dirichlet noise
  uintf cur_dirichlet = 0;
  for (uintf i = 0; i < NUM_MOVES; ++i) {
    if (cur_node->is_legal(i)) {
      p[i] = p[i] * scalar + dirichlet_noise[cur_dirichlet] * dirichlet_scalar;
      ++cur_dirichlet;
    }
  }

  delete dirichlet_noise;

  for (uintf i = 0; i < NUM_TOTAL_MOVES; ++i) {
    cur_node->set_probability(i, (unsigned short)lround(p[i] * 511.0));
  }
  // Propagate evaluation
  // Set evaluation for first node? Must work with terminal evals as well
  float cur_evaluation = evaluation;
  while (cur != root) {
    cur_node->add_evaluation(cur_evaluation);
    cur_evaluation *= -1.0;
    cur = cur_node->get_parent();
    cur_node = trainer->get_node(cur);
  }
  // Propagate to root
  cur_node->add_evaluation(cur_evaluation);
}

// Figure out what the Python code does for this for the first search of a move,
// where evaluation is not needed We can overload, factor out the search, add
// "receive_evaluation" function
bool TrainMC::search(float game_state[GAME_STATE_SIZE]) {

  bool need_evaluation = false;

  while (!need_evaluation && iterations_done < max_iterations) {

    ++iterations_done;

    while (!cur_node->is_terminal()) {
      cur_node->increment_visits();
      uintf move_choice = choose_next();
      // Exploring a new node
      if (!(cur_node->has_visited(move_choice))) {
        cur_node->set_visit(move_choice);
        // Create the new node
        cur = trainer->place_next(cur_node->get_seed(), cur_node->get_game(),
                                  cur_node->get_depth(), cur, move_choice);
        cur_node = trainer->get_node(cur);
        break;
      }
      // Otherwise, move down normally
      else {
        cur = trainer->find_next(cur_node->get_seed(), move_choice);
        cur_node = trainer->get_node(cur);
      }
    }

    // Check for terminal state, otherwise evaluation is needed
    if (cur_node->is_terminal()) {
      // Otherwise visits is not incremented
      cur_node->increment_visits();
      // Don't propagate if value is 0
      if (cur_node->get_result() != DRAW) {
        // Propagate evaluation
        // In a decisive terminal state, the person to play is always the loser
        // It is important we don't add to visits here, because we skip this for
        // draws
        float cur_evaluation = -1.0;
        while (cur != root) {
          cur_node->add_evaluation(cur_evaluation);
          cur_evaluation *= -1.0;
          cur = cur_node->get_parent();
          cur_node = trainer->get_node(cur);
        }
        // We need to propagate to root (do we?)
        cur_node->add_evaluation(cur_evaluation);
      } else {
        cur = root;
        cur_node = trainer->get_node(cur);
      }
    }
    // Otherwise, request an evaluation
    else {
      cur_node->write_game_state(game_state);
      need_evaluation = true;
    }
  }

  return need_evaluation;
}

uintf TrainMC::choose_next() {

  // Random value lower than -1.0
  float max_value = -2.0;
  // Initialize this variable to be safe
  uintf move_choice = 0;

  for (uintf i = 0; i < NUM_MOVES; ++i) {
    float u;
    // Check if it is a legal move
    if (cur_node->is_legal(i)) {
      if (cur_node->has_visited(i)) {
        Node *next =
            trainer->get_node(trainer->find_next(cur_node->get_seed(), i));
        u = -1.0 * next->get_evaluation() / (float)next->get_visits() +
            c_puct * cur_node->get_probability(i) *
                sqrt((float)cur_node->get_visits() - 1.0) /
                ((float)next->get_visits() + 1);

      }
      // If not visited, set action value to 0
      else {
        u = c_puct * cur_node->get_probability(i) *
            sqrt((float)cur_node->get_visits() - 1.0);
      }
      // We assume there are no ties
      if (u > max_value) {
        max_value = u;
        move_choice = i;
      }
    }
  }

  return move_choice;
}