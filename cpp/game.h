#ifndef GAME_H
#define GAME_H

// Bitset can save on memory and make resetting all bits faster
#include <bitset>
#include <memory>

using std::bitset;
using std::shared_ptr;

class Game {

    bitset<48> board;
    bitset<16> frozen;
    uint_least32_t pieces:18, outcome:2, to_play:1;
  
  public:

    Game();
    bool is_empty(uint_fast8_t, uint_fast8_t);
    bool can_place(uint_fast8_t, uint_fast8_t);
    bool can_move(uint_fast8_t, uint_fast8_t);
    bool is_legal(uint_fast8_t);
    bitset<96> get_legal_moves();

};

class Node {
    Game game;
    int searches, depth;
    Node *parent;
    bitset<96> legal_moves;
    float noisy_probabilities[96];
    float evaluations[96];
    shared_ptr<Node> children[96];
};
#endif
