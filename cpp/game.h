#ifndef GAME_H
#define GAME_H

// Bitset can save on memory and make resetting all bits faster
#include "util.h"
#include <bitset>
#include <memory>
#include <cstdint>

using std::bitset;

const uint BOARD_SIZE = 16;

class Game {

    bitset<3*BOARD_SIZE> board;
    // Is it better to store this as 1/2 ints?
    bitset<BOARD_SIZE> frozen;
    // We want to add orientation later
    uint pieces[6], to_play;
    // Game result, Game needs to have it because it knows which lines exist
    Result result;

    bool is_empty(uint row, uint col);
    bool can_place(uint ptype, uint row, uint col);
    int8 get_bottom(uint row, uint col);
    int8 get_top(uint row, uint col);
    bool can_move(uint row1, uint col1, uint row2, uint col2);
    bool is_legal(uint move_id);
    bool get_line_breakers(bitset<NUM_MOVES> &legal_moves);

  public:

    Game();
    Game(const Game &game) = default;
    ~Game() = default;

    void get_legal_moves(bitset<NUM_MOVES> &legal_moves);
    void do_move(uint);
    bool is_terminal();
    void write_game_state(float game_state[GAME_STATE_SIZE])

};

#endif
