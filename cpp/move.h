#ifndef MOVE_H
#define MOVE_H

#include "util.h"

struct Move {

    bool mtype;
    uint ptype, row1, col1, row2, col2;

    Move(uint);

};

uint encode_place(uint ptype, uint row, uint col);
uint encode_move(uint row1, uint col1, uint row2, uint col2);

#endif
