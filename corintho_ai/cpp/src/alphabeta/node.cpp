#include "alphabeta/node.h"
#include "game.h"
#include <vector>

namespace AlphaBeta {

Node::Node() {
  score = Value();
  previousScore = Value();
  alpha = Value(Value::ValueType::NEGATIVE_INFINITY);
  beta = Value(Value::ValueType::POSITIVE_INFINITY);
  children = std::vector<Node>();
  gameState = Game();
  initialized = false;
}

Node::~Node() {
    children.clear();
}


void Node::createChildren() {
    if (initialized) {
        return;
    }
    std::bitset<kNumMoves> legal_moves;
    gameState.getLegalMoves(legal_moves);
    for (int i = 0; i < kNumMoves; i++) {
        if (legal_moves[i]) {
            Node child = Node();
            child.gameState = gameState;
            child.gameState.doMove(i);
            children.push_back(child);
        }
    }
  initialized = true;
}

Value Node::evaluate() {
  this->createChildren();
  return children.size();
}

Value Node::search(int depth, Value alpha, Value beta) {
  this->createChildren();
  if (depth == 0 || children.empty()) {
    return evaluate();
  }
  Value searchValue = AlphaBeta::Value(AlphaBeta::Value::ValueType::NEGATIVE_INFINITY);
  for (Node &child : children) {
    searchValue = std::max(searchValue,-child.search(depth - 1, -beta, -alpha));
    if (searchValue >= beta) {
      score = beta;
      return beta;
    }
    if (searchValue > alpha) {
      alpha = searchValue;
    }
  }
  score = alpha;
  return alpha;
}

}  // namespace AlphaBeta
