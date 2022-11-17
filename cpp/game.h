#ifndef GAME_H
#define GAME_H

// Bitset can save on memory and make resetting all bits faster
#include "util.h"
#include <bitset>
#include <memory>
#include <cstdint>

using std::bitset;

const uint8 BOARD_SIZE = 16;

class Game {

    bitset<3*BOARD_SIZE> board;
    // Is it better to store this as 1/2 ints?
    bitset<BOARD_SIZE> frozen;
    // We want to add orientation later
    uint8 pieces[6], to_play;
    int8 outcome;

    bool is_empty(uint8 row, uint8 col);
    bool can_place(uint8 ptype, uint8 row, uint8 col);
    int8 get_bottom(uint8 row, uint8 col);
    int8 get_top(uint8 row, uint8 col);
    bool can_move(uint8 row1, uint8 col1, uint8 row2, uint8 col2);
    bool is_legal(uint8 move_id);
    void get_line_breakers(bitset<NUM_MOVES> &legal_moves);

  public:

    Game();
    Game(const Game &game) = default;
    void get_legal_moves(bitset<NUM_MOVES> &legal_moves);
    void do_move(int);

};

#endif
