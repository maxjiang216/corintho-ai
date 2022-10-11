#ifndef MOVE_H
#define MOVE_H
#include "stdbool.h"

struct Move {
    bool mtype;
    short row1, col1, row2, col2, ptype;
};

struct Move move_from_id(short move_id);
short encode_place(short ptype, short row, short col);
short encode_move(short row1, short col1, short row2, short col2);

#endif
