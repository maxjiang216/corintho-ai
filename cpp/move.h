#ifndef MOVE_H
#define MOVE_H

#include "util.h"

struct Move {

    bool mtype;
    uint8 ptype, row1, col1, row2, col2;

    Move(uint8);

};

uint8 encode_place(uint8 ptype, uint8 row, uint8 col);
uint8 encode_move(uint8 row1, uint8 col1, uint8 row2, uint8 col2);

#endif
