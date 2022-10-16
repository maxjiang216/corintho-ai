#ifndef MOVE_H
#define MOVE_H

struct Move {

    bool mtype;
    int ptype, row1, col1, row2, col2;

    Move() = default;
    Move(int);
    Move(int, int, int);
    Move(int, int, int, int);

};

int encode_place(int, int, int);
int encode_move(int, int, int, int);

#endif
