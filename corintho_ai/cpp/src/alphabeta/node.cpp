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
    // std::cout << "Game State:\n" << gameState << std::endl;
    // std::cout << "Number of Children Created: " << children.size() << std::endl;
  initialized = true;
}

Value Node::evaluate() {
  static std::random_device rd; 
  static std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-0.01, 0.01);
  double randomComponent = dis(gen);
  if (initialized) {
    return Value(children.size()+randomComponent);
  }
  std::bitset<kNumMoves> legal_moves;
  gameState.getLegalMoves(legal_moves);
  return Value(legal_moves.count()+randomComponent);
}

Value Node::search(int depth, Value alpha, Value beta) {
  this->createChildren();
  if (depth == 0 || children.empty()) {
    Value value = Value(evaluate());
    // std::cout << "Depth: " << depth << std::endl;
    // std::cout << "Value: " << value << std::endl;
    // std::cout << "Game State:\n" << gameState << std::endl;
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
  // std::cout << "Depth: " << depth << std::endl;
  // std::cout << "Score: " << score << std::endl;
  // std::cout << "Game State:\n" << gameState << std::endl;
  return alpha;
}

Value Node::getScore() const {
  return score;
}

std::vector<Node>& Node::getChildren() {
  return children;
}

const Game& Node::getGameState() const {
  return gameState;
}

void Node::sortChildrenRecursively() {
  std::stable_sort(children.begin(), children.end(), [](const Node& a, const Node& b) {
    return a.getScore() > b.getScore();
  });

  // Recursively sort the children of each child
  for (Node& child : children) {
    child.sortChildrenRecursively();
  }
}

}  // namespace AlphaBeta
