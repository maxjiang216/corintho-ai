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
    unsigned short pieces[6], to_play, orientation;
    short outcome;

    bool is_empty(int, int);
    bool can_place(int, int, int);
    int get_bottom(int, int);
    int get_top(int, int);
    bool can_move(int, int, int, int);
    bool is_legal(int);
    void get_line_breakers(bitset<96> &legal_moves);

  public:

    Game();
    Game(const Game &game) = default;
    void get_legal_moves(bitset<96> &legal_moves);
    void do_move(int);

};

#endif
