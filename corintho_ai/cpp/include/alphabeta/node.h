#ifndef ALPHABETA_NODE_H
#define ALPHABETA_NODE_H

#include "alphabeta/dtypes.h"
#include "game.h"
#include "move.h"
#include <vector>
#include <optional>

namespace AlphaBeta {

class Node {
 public:
  Node();
  ~Node();
  void createChildren();
  Value evaluate();
  Value search(int depth, Value alpha, Value beta);

  Value getScore() const;
  const std::vector<Node>& getChildren() const;
  const Game& getGameState() const;
  std::optional<int> getMove() const;
  void sortChildrenRecursively();
  std::optional<Move> getBestMove() const;
  

 private:
  Value score;
  Value previousScore;
  Value alpha;
  Value beta;
  std::optional<int> move;
  std::vector<Node> children;
  Game gameState;
  bool initialized;
};

}  // namespace AlphaBeta

#endif