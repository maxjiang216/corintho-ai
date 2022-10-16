#ifndef MOVE_H
#define MOVE_H

class Move {

  public:

    Move();
    Move(short, short, short);
    Move(short, short, short, short);

    bool mtype;
    short row1, col1, row2, col2, ptype;
};

Move get_move_from_id(short);
short encode_place(short, short, short);
short encode_move(short, short, short, short);

#endif
