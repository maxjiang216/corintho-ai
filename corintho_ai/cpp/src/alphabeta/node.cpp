#include "alphabeta/node.h"
#include "game.h"
#include <vector>
#include <algorithm>
#include <iostream>
#include <random>
#include <chrono>

namespace AlphaBeta {

Node::Node() {

  score = Value();
  previousScore = Value();
  alpha = Value(Value::ValueType::NEGATIVE_INFINITY);
  beta = Value(Value::ValueType::POSITIVE_INFINITY);
  children = std::vector<Node>();
  gameState = Game();
  move = std::nullopt;
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
            child.move = i;
            child.gameState.doMove(i);
            children.push_back(child);
        }
    }
  initialized = true;
}

Value Node::evaluate() {
  std::bitset<kNumMoves> legal_moves;
  bool is_lines = gameState.getLegalMoves(legal_moves);
  if (legal_moves.count() == 0) {
    if (is_lines) {
      return Value(-300);
    }
    return Value(0);
  }
  static std::random_device rd; 
  static std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-0.01, 0.01);
  double randomComponent = dis(gen);
  if (initialized) {
    return Value(children.size()+randomComponent);
  }
  return Value(legal_moves.count()+randomComponent);
}

Value Node::search(int depth, Value alpha, Value beta) {
  this->createChildren();
  if (depth == 0 || children.empty()) {
    Value value = Value(evaluate());
    score = value;
    return value;
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

Value Node::getScore() const {
  return score;
}

const std::vector<Node>& Node::getChildren() const {
  return children;
}

const Game& Node::getGameState() const {
  return gameState;
}

std::optional<int> Node::getMove() const {
  return move;
}

void Node::sortChildrenRecursively() {
  std::stable_sort(children.begin(), children.end(), [](const Node& a, const Node& b) {
    return a.getScore() < b.getScore();
  });

  // Recursively sort the children of each child
  for (Node& child : children) {
    child.sortChildrenRecursively();
  }
}

std::optional<Move> Node::getBestMove() const {
  if (children.empty()) {
    return std::nullopt;
  }
  if (children[0].getMove().has_value()) {
    return Move(children[0].getMove().value());
  }
  return std::nullopt;
}

}  // namespace AlphaBeta
