#ifndef GAME_H
#define GAME_H

// Bitset can save on memory and make resetting all bits faster
#include <bitset>

using std::bitset;

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
    Game *game;
    uint_least16_t searches:12, visits:12;
    float evaluation;
    uint_least8_t depth:6;
    Node *parent;
    bitset<96> legal_moves;
    //float noisy_probabilities[96];
    //Node *children[96];
};
#endif
