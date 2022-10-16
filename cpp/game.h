#ifndef GAME_H
#define GAME_H

// Bitset can save on memory and make resetting all bits faster
#include <bitset>

using std::bitset;

class Game {

    bitset<48> board;
    bitset<16> frozen;
    bool is_done;
    int_least8_t to_play, pieces[6], outcome;
  
  public:

    Game();
    bool is_empty(int_least8_t, int_least8_t);
    bool can_place(int_least8_t, int_least8_t);
    bool can_move(int_least8_t, int_least8_t);
    bool is_legal(int_least8_t);

};
#endif
