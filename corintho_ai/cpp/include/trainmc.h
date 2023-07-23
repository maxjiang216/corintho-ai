#ifndef TRAINMC_H
#define TRAINMC_H

#include <random>
#include <vector>

#include "util.h"

class Game;
class Node;

/// @brief Class for Monte Carlo tree search
class TrainMC {
 public:
  /// @brief Constructor
  TrainMC(std::mt19937 *generator, float *to_eval, int32_t max_searches = 1600,
          int32_t searches_per_eval = 16, float c_puct = 1.0,
          float epsilon = 0.25, bool testing = false);
  TrainMC(const TrainMC &) = default;
  TrainMC(TrainMC &&) = default;
  TrainMC &operator=(const TrainMC &) = default;
  TrainMC &operator=(TrainMC &&) = default;
  ~TrainMC();
  /// @brief Constructor for web app. Starts at an arbitrary position
  TrainMC(std::mt19937 *generator, float *to_eval, int32_t max_searches,
          int32_t searches_per_eval, float c_puct, float epsilon,
          int32_t board[4 * kBoardSize], int32_t to_play, int32_t pieces[6]);

  /// @brief Return the root node of the Monte Carlo search tree
  Node *root() const noexcept;
  /// @brief Returns the evaluation of the root node
  /// @details This is used in the web app.
  float eval() const noexcept;
  /// @brief Returns the number of requests for evaluations
  int32_t num_requests() const noexcept;
  /// @brief Returns if there are no evaluations requested
  bool no_requests() const noexcept;
  /// @brief Returns if the tree is uninitialized
  bool uninitialized() const noexcept;
  /// @brief Returns the number of nodes in the tree
  int32_t num_nodes() const noexcept;
  /// @brief Returns if the game is done.
  /// @details Returns if the root node is a terminal position. This is used in
  /// the web app.
  bool done() const noexcept;
  /// @brief Returns if the game is drawn. This is used in the web app.
  bool drawn() const noexcept;

  /// @brief Set the root node to have the given game and depth
  void null_root() noexcept;

  /// @brief Write the game states for the positions we need to evaluate
  /// @details This is used when playing with the AI.
  void writeRequests(float *game_states) const noexcept;
  /// @brief Get the legal moves of the root node.
  /// @details This is used for the web app.
  void getLegalMoves(int32_t legal_moves[kNumMoves]) const noexcept;

  /// @brief Find best move and move the root node to that node. Writes the
  /// game state and probability samples for training.
  /// @return The ID of the best move
  int32_t chooseMove(float game_state[kGameStateSize] = nullptr,
                     float prob_sample[kNumMoves] = nullptr) noexcept;
  /// @brief Do an iteration of searches
  /// @param eval The evaluations for the positions requested.
  /// For the first search, this is nullptr
  /// @param probs The probabilities for legal moves for the positions
  /// requested For the first search, this is nullptr
  /// @return Whether the turn is done. This happens when the number of
  /// searches equals TrainMC::max_iterations_ Or when the root node's outcome
  /// is known.
  /// @details This function will perform searches until the number of
  /// positions where an evaluation is requested equals
  /// TrainMC::searches_per_eval_, when the root node's outcome is known, or
  /// when the root node's children are completely searched.
  bool doIteration(float eval[] = nullptr, float probs[] = nullptr);
  /// @brief Receive the opponent's move and move the root node to that node
  /// @return Whether the move was searched and thus does not need an
  /// evaluation
  /// @details Will copy game and depth from the opponent if the move has not
  /// been searched
  bool receiveOpponentMove(int32_t move_choice, const Game &game,
                           int32_t depth);
  /// @brief Create a new root node with the given game and depth
  /// @details This is used when the opponent makes an unsearched move. Does
  /// not delete the old root (if it exists).
  void createRoot(const Game &game, int32_t depth);

 private:
  /// @brief The output of chooseNext
  struct ChooseNextOutput {
    enum class Type { kVisited, kNew, kNone } type;
    int32_t choice;
    Node *node;
  };
  /// @brief Apply legal move filter and normalize
  void getFilteredProbs(float probs[kNumMoves],
                        float filtered_probs[]) const noexcept;
  /// @brief Generate Dirichlet noise
  void generateDirichlet(float dirichlet[]) const noexcept;

  /// @brief Set integer probabilities to edges, sets denominator of cur_ node.
  void setProbs(float filtered_probs[], float dirichlet[]) noexcept;
  /// @brief Write the neural network outputs into the node
  void receiveEval(float eval[], float probs[]) noexcept;
  /// @brief Choose a move based on the probabilities of the root node.
  /// @details This is mostly used for one-search strategies.
  int32_t chooseHighProbMove() const noexcept;
  /// @brief Choose move for when the root node is a winning position
  int32_t chooseMoveWon(float prob_sample[kNumMoves]) noexcept;
  /// @brief Choose move for when the root node is a losing or drawn position
  /// @details The logic in these cases is the same except drawn positions will
  /// avoid choosing losing moves
  int32_t chooseMoveLostDrawn(float prob_sample[kNumMoves]) noexcept;
  /// @brief Choose a move for the opening moves
  /// @details Temperate is 1.
  int32_t chooseMoveOpening(float prob_sample[kNumMoves]) noexcept;
  /// @brief Choose a move for the standard case
  int32_t chooseMoveNormal(float prob_sample[kNumMoves]) noexcept;
  /// @brief Move down the Monte Carlo search tree
  /// @details This occurs when we choose a move.
  void moveDown(Node *prev) noexcept;
  /// @brief Propagate results of terminal nodes and deduced results
  /// @details We do this each time a terminal node is searched.
  /// Although it may potentially propagate results all the way up the tree, on
  /// average this operation is relatively cheap, especially compared to neural
  /// network evaluations. We use elementary game theory to deduce the results
  /// of nodes that are not terminal.
  void propagateTerminal() noexcept;
  /// @brief Choose the next node in the Monte Carlo search
  /// @return The ID of the move to take or -1 if no move is available
  /// @details Sets cur_ to a child node of cur_.
  ChooseNextOutput chooseNext() noexcept;
  /// @brief Repeated move down the Monte Carlo search tree until a terminal or
  /// unsearched node is reached
  /// @return Whether our search is done. This occurs if we have done the
  /// maximum number of searches, if we have deduced the result of the root
  /// node, or if all the possible new nodes have been searched.
  /// @details Uses UCB to search the best edge to take in a Monte Carlo search
  /// tree
  void search();

  /// @brief The number of initial moves to consider "opening" moves
  /// @details When choosing a move during the opening, temperature is 1
  /// Compared to elsewhere where it is 0. This is a hyperparameter.
  /// Note that the average Corintho game lasts about 30 moves.
  static constexpr int32_t kNumOpeningMoves = 6;

  /// @brief The root node of the Monte Carlo search tree
  Node *root_{nullptr};
  /// @brief The current node we are at when searching the Monte Carlo search
  /// tree
  Node *cur_{nullptr};
  /// @brief The number of searches done for the current move
  int32_t searches_done_{0};
  /// @brief The maximum number of searches to do per move.
  /// @details 1600 was used during training.
  const int32_t max_searches_{1600};
  /// @brief The number of searches to do per neural network evaluation
  /// @details The actual number of searches done can be greater than this
  /// since some searches do not require an evaluation and are not counted.
  /// Evaluating many positions at the same time leverages parallelism in the
  /// neural network. 16 was used during training.
  const int32_t searches_per_eval_{16};
  /// @brief c_puct in the UCB formula
  /// @details 1.0 was used during training.
  const float c_puct_{1.0};
  /// @brief The weight of Dirichlet noise compared to neural network
  /// probabilities
  /// @details 0.25 was used during training.
  const float epsilon_{0.25};
  /// @brief The nodes we have searched this cycle that need to be evaluated
  std::vector<Node *> searched_{};
  /// @brief The location to write the game position that needs to be evaluated
  /// @details This is shared between the two players in a SelfPlayer,
  /// since only one player is searching at a time.
  float *to_eval_{nullptr};
  /// @brief Whether this is testing mode or training mode
  /// @details In testing mode, we do not use temperature 1 in the opening
  /// and we also do not write training samples.
  bool testing_{false};
  /// @brief The random generator for all operations
  /// @details This is shared between the two players in a SelfPlayer,
  /// since only one player is searching at a time.
  std::mt19937 *generator_{nullptr};
};

#endif
