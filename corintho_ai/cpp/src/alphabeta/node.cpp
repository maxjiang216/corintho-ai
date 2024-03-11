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
  return Value(0);
}

Value Node::search(int depth, Value alpha, Value beta) {
  if (depth == 0 || children.empty()) {
    return evaluate();
  }

  for (Node &child : children) {
    Value score = -child.search(depth - 1, -beta, -alpha);
    if (score >= beta) {
      return beta;
    }
    if (score > alpha) {
      alpha = score;
    }
  }
  return alpha;
}

}  // namespace AlphaBeta
