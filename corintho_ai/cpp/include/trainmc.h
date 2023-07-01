#ifndef TRAINMC_H
#define TRAINMC_H

#include <random>
#include <vector>

class Game;
class Node;

class TrainMC {
 public:

  // @brief Constructor
  TrainMC(std::mt19937 *generator, bool testing=false) noexcept;

  // @brief Return the root node of the Monte Carlo search tree
  Node *root() const noexcept;
  // @brief Return the depth of the root node
  int32_t root_depth() const noexcept;
  // @brief Return a reference to the game of the root node
  const Game& get_root_game() const noexcept;

  // @brief Return if there are no evaluations requested
  bool noEvalsRequested() const noexcept;
  // @brief Return the number of nodes searched since the last evaluation
  int32_t numNodesSearched() const noexcept;
  // @brief Return if the tree is uninitialized
  bool isUninitialized() const noexcept;
  // @brief Return the number of nodes in the tree
  int32_t numNodes() const noexcept;

  // @brief Set the root node to have the given game and depth
  void createRoot(const Game &game, int32_t depth);

  // @brief Do an iteration of searches
  // @param evaluation The evaluations for the positions requested.
  // For the first search, this is nullptr
  // @param probabilities The probabilities for legal moves for the positions requested
  // For the first search, this is nullptr
  // @return Whether the turn is done. This happens when the number of searches equals TrainMC::max_iterations_
  // Or when the root node's outcome is known.
  // @details This function will perform searches until the number of positions where an evaluation is requested
  // equals TrainMC::searches_per_eval_, when the root node's outcome is known,
  // or when the root node's children are completely searched.
  bool doIteration(float evaluation[]=nullptr, float probabilities[]=nullptr);

  // Choose the next child to visit
  int32_t chooseMove(float game_state[kGameStateSize],
                    float probability_sample[kNumMoves]) noexcept;

  bool receiveOpponentMove(uintf move_choice, const Game &game, uintf depth);

private:

  static void set_max_iterations(uintf new_max_iterations) noexcept;
  static void set_c_puct(float new_c_puct) noexcept;
  static void set_epsilon(float new_epsilon) noexcept;
  static void set_searches_per_eval(uintf new_searches_per_eval) noexcept;

  // I think we have to initialize these
  inline static uintf max_iterations = 1600, searches_per_eval = 1;
  inline static float c_puct = 1.0, epsilon = 0.25;
  // Number of moves to use weighted random
  const uintf NUM_OPENING_MOVES = 6;

  Node *root, *cur;

  // Which index to write to for the searched node
  uintf eval_index;
  // Nodes we have searches this cycle
  std::vector<Node *> searched;
  // We want to evaluate these vectors
  float *to_eval;

  // Used to keep track of when to choose a move
  uintf iterations_done;

  bool testing, logging;

  std::mt19937 *generator;

  void receive_evaluation(float evaluation[], float probabilities[]);
  bool search();

  void move_down(Node *prev_node);

  void propagate_result();

};

#endif
