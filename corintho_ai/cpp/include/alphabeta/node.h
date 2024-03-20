#ifndef ALPHABETA_NODE_H
#define ALPHABETA_NODE_H

#include "alphabeta/dtypes.h"
#include "game.h"
#include <vector>

namespace AlphaBeta {

class Node {
 public:
  Node();
  ~Node();
  void createChildren();
  Value evaluate();
  Value search(int depth, Value alpha, Value beta);

  Value getScore() const;
  std::vector<Node>& getChildren();
  const Game& getGameState() const;
  void sortChildrenRecursively();
  

 private:
  Value score;
  Value previousScore;
  Value alpha;
  Value beta;
  std::vector<Node> children;
  Game gameState;
  bool initialized;
};

}  // namespace AlphaBeta

#endif