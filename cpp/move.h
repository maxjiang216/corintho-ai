#ifndef MOVE_H
#define MOVE_H

struct Move {

    bool mtype;
    uint_fast8_t ptype, row1, col1, row2, col2;

    Move(uint_fast8_t);

};

uint_fast8_t encode_place(uint_fast8_t, uint_fast8_t, uint_fast8_t);
uint_fast8_t encode_move(uint_fast8_t, uint_fast8_t, uint_fast8_t);

#endif
