#ifndef MOVE_H
#define MOVE_H

struct Move {

    bool mtype:1;
    uint_fast8_t ptype:2, row1:2, col1:2, row2:2, col2:2;

    Move(uint_fast8_t);

};

uint_fast8_t encode_place(uint_fast8_t, uint_fast8_t, uint_fast8_t);
uint_fast8_t encode_move(uint_fast8_t, uint_fast8_t, uint_fast8_t);

#endif
