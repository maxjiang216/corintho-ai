#ifndef TRAINMC_H
#define TRAINMC_H

#include <random>
#include <vector>

class Game;
class Node;

class TrainMC {
 public:
  // @brief Constructor
  TrainMC(std::mt19937 *generator, int32_t max_searches = 1600,
          int32_t searches_per_eval = 16, float c_puct = 1.0,
          float epsilon = 0.25, bool testing = false);

  // @brief Return the root node of the Monte Carlo search tree
  Node *root() const noexcept;
  // @brief Return the depth of the root node
  int32_t root_depth() const noexcept;
  // @brief Return a reference to the game of the root node
  const Game &get_root_game() const noexcept;

  // @brief Return if there are no evaluations requested
  bool noEvalsRequested() const noexcept;
  // @brief Return the number of nodes searched since the last evaluation
  // TODO Check if this is used
  int32_t numNodesSearched() const noexcept;
  // @brief Return if the tree is uninitialized
  bool isUninitialized() const noexcept;
  // @brief Return the number of nodes in the tree
  int32_t numNodes() const noexcept;

  // @brief Set the root node to have the given game and depth
  void createRoot(const Game &game, int32_t depth);

  // @brief Find best move and move the root node to that node. Writes the game
  // state and probability samples for training.
  // @return The ID of the best move
  int32_t chooseMove(float game_state[kGameStateSize],
                     float prob_sample[kNumMoves]) noexcept;
  // @brief Do an iteration of searches
  // @param eval The evaluations for the positions requested.
  // For the first search, this is nullptr
  // @param probs The probabilities for legal moves for the positions requested
  // For the first search, this is nullptr
  // @return Whether the turn is done. This happens when the number of searches
  // equals TrainMC::max_iterations_ Or when the root node's outcome is known.
  // @details This function will perform searches until the number of positions
  // where an evaluation is requested equals TrainMC::searches_per_eval_, when
  // the root node's outcome is known, or when the root node's children are
  // completely searched.
  bool doIteration(float eval[] = nullptr, float probs[] = nullptr);
  // @brief Receive the opponent's move and move the root node to that node
  // @return Whether the move was searched and thus does not need an evaluation
  // @details Will copy game and depth from the opponent if the move has not
  // been searched
  bool receiveOpponentMove(int32_t move_choice, const Game &game,
                           int32_t depth);

 private:
  // @brief Write the neural network outputs into the node
  void receiveEval(float eval[], float probs[]) noexcept;
  // @brief Move down the Monte Carlo search tree
  // @details This occurs when we choose a move.
  void moveDown(Node *prev_node) noexcept;
  // @brief Propagate results of terminal nodes and deduced results
  // @details We do this each time a terminal node is searched.
  // Although it may potentially propagate results all the way up the tree, on
  // average this operation is relatively cheap, especially compared to neural
  // network evaluations. We use elementary game theory to deduce the results of
  // nodes that are not terminal.
  void propagateTerminal() noexcept;
  // @brief Repeated move down the Monte Carlo search tree until a terminal or
  // unsearched node is reached
  // @return Whether our search is done. This occurs if we have done the maximum
  // number of searches, if we have deduced the result of the root node, or if
  // all the possible new nodes have been searched.
  // @details Uses UCB to search the best edge to take in a Monte Carlo search
  // tree
  bool search();

  // @brief The number of initial moves to consider "opening" moves
  // @details When choosing a move during the opening, temperature is 1
  // Compared to elsewhere where it is 0. This is a hyperparameter.
  // Note that the average Corintho game lasts about 30 moves.
  static constexpr int32_t kNumOpeningMoves = 6;

  // @brief The root node of the Monte Carlo search tree
  Node *root_{nullptr};
  // @brief The current node we are at when searching the Monte Carlo search
  // tree
  Node *cur_{nullptr};
  // @brief The number of searches done for the current move
  int32_t searches_done_{0};
  // @brief The maximum number of searches to do per move.
  // @details 1600 was used during training.
  const int32_t max_searches_{1600};
  // @brief The number of searches to do per neural network evaluation
  // @details The actual number of searches done can be greater than this
  // since some searches do not require an evaluation and are not counted.
  // Evaluating many positions at the same time leverages parallelism in the
  // neural network. 16 was used during training.
  const int32_t searches_per_eval_{16};
  // @brief c_puct in the UCB formula
  // @details 1.0 was used during training.
  const float c_puct_{1.0};
  // @brief The weight of Dirichlet noise compared to neural network
  // probabilities
  // @details 0.25 was used during training.
  const float epsilon_{0.25};
  // @brief The nodes we have searched this cycle that need to be evaluated
  std::vector<Node *> searched_{};
  // @brief The index to put the next searched node into
  int32_t searched_index_{0};
  // @brief The location to write the game position that needs to be evaluated
  // @details This is shared between the two players in a SelfPlayer,
  // since only one player is searching at a time.
  float *to_eval_{nullptr};
  // @brief Whether this is testing mode or training mode
  // @details In testing mode, we do not use temperature 1 in the opening
  // and we also do not write training samples.
  bool testing_{false};
  // @brief The random generator for all operations
  // @details This is shared between the two players in a SelfPlayer,
  // since only one player is searching at a time.
  std::mt19937 *generator_{nullptr};
};

#endif
