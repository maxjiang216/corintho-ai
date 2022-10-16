#ifndef GAME_H
#define GAME_H

// Bitset can save on memory and make resetting all bits faster
#include <bitset>

using std::bitset;

class Game {

    bitset<48> board;
    bitset<16> frozen;
    bool is_done;
    uint_least8_t to_play, pieces[6];
    int_least8_t outcome;
  
  public:

    Game();
    bool is_empty(uint_fast8_t, uint_fast8_t);
    bool can_place(uint_fast8_t, uint_fast8_t);
    bool can_move(uint_fast8_t, uint_fast8_t);
    bool is_legal(uint_fast8_t);
    bitset<96> get_legal_moves();

};
#endif
