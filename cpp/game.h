#ifndef GAME_H
#define GAME_H

// Bitset can save on memory and make resetting all bits faster
#include "util.h"
#include <bitset>
#include <ostream>

using std::bitset;

const uintf BOARD_SIZE = 16;

class Game {

    bitset<3*BOARD_SIZE> board;
    bitset<BOARD_SIZE> frozen;
    // We want to add orientation later
    uintf to_play, pieces[6];
    bitset<NUM_MOVES> legal_moves;
    // Game result
    // Game needs to have it because it knows which lines exist
    Result result;

    void get_legal_moves();
    // Finds lines and moves that break all lines
    // Returns whether there were lines
    bool get_line_breakers();

    // Returns an int representing the top of a stack, -1 if empty
    intf get_top(uintf row, uintf col) const;
    // Returns an int representing the bottom piece of a stack, 3 if empty
    intf get_bottom(uintf row, uintf col) const;

    void apply_line(uintf line);

    bool is_legal_move(uintf move_id) const;
    bool can_place(uintf ptype, uintf row, uintf col) const;
    bool can_move(uintf row1, uintf col1, uintf row2, uintf col2) const;
    bool is_empty(uintf row, uintf col) const;

  public:

    Game();
    Game(const Game &game) = default;
    ~Game() = default;

    // Accessors
    bool is_legal(uintf move_choice) const;
    uintf get_to_play() const;
    Result get_result() const;
    bool is_terminal() const;

    void do_move(uintf move_id);
    
    void write_game_state(float game_state[GAME_STATE_SIZE]) const;

    friend std::ostream& operator<<(std::ostream& stream, const Game &game);

};

#endif
