#ifndef GAME_H
#define GAME_H

// Bitset can save on memory and make resetting all bits faster
#include "util.h"
#include <bitset>
#include <memory>
#include <cstdint>

using std::bitset;

const uintf BOARD_SIZE = 16;

class Game {

    bitset<3*BOARD_SIZE> board;
    // Is it better to store this as 1/2 ints?
    bitset<BOARD_SIZE> frozen;
    // We want to add orientation later
    uintf pieces[6], to_play;
    // Game result, Game needs to have it because it knows which lines exist
    Result result;

    bool is_empty(uintf row, uintf col);
    bool can_place(uintf ptype, uintf row, uintf col);
    intf get_bottom(uintf row, uintf col);
    intf get_top(uintf row, uintf col);
    bool can_move(uintf row1, uintf col1, uintf row2, uintf col2);
    bool is_legal(uintf move_id);
    bool get_line_breakers(bitset<NUM_MOVES> &legal_moves);

  public:

    Game();
    Game(const Game &game) = default;
    ~Game() = default;

    void get_legal_moves(bitset<NUM_MOVES> &legal_moves);
    void do_move(uintf move_id);
    bool is_terminal();
    Result get_result();
    uintf get_to_play();
    void write_game_state(float game_state[GAME_STATE_SIZE]);

};

#endif
