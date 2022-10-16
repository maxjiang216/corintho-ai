#ifndef GAME_H
#define GAME_H

// Bitset can save on memory and make resetting all bits faster
#include <bitset>

using std::bitset;

class Game {

    bitset<48> board;
    bitset<16> frozen;
    bool is_done;
    int to_play, pieces[6], outcome;
  
  public:

    Game();
    bool is_empty(int, int);
    bool can_place(int, int, int);
    bool can_move(int, int, int, int);
    bool is_legal(int);

};
#endif
