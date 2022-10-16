#ifndef MOVE_H
#define MOVE_H

struct Move {

    bool mtype;
    int_least8_t ptype, row1, col1, row2, col2;

    Move(int_least8_t);

};

int_least8_t encode_place(int_least8_t, int_least8_t, int_least8_t);
int_least8_t encode_move(int_least8_t, int_least8_t, int_least8_t);

#endif
