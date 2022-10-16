#include <stdlib.h>
#include "move.h"

// 3 int constructor
// Used for place moves
Move::Move(int ptype, int row, int col): mtype{true}, ptype{ptype}, row1{row}, col1{col} {}

// 4 int constructor
// Used for move moves
Move::Move(int row1, int col1, int row2, int col2): mtype{false}, row1{row1}, col1{col1}, row2{row2}, col2{col2} {}

// Convert move_id to Move object
Move get_move_from_id(int move_id) {

    // Place
    // I believe that place is more common
    if (move_id >= 48) {
	return Move{(move_id - 48) / 16, (move_id % 16) / 4, move_id % 4};
    }

    // Move

    // Right
    if (move_id < 12)
        return Move{move_id / 3, move_id % 3, move_id / 3, move_id % 3 + 1};

    // Down
    if (move_id < 24)
	return Move{(move_id - 12) / 4, move_id % 4, (move_id - 8) / 4, move_id % 4};

    // Left
    if (move_id < 36)
	return Move{(move_id - 24) / 3, move_id % 3 + 1, (move_id - 24) / 3, move_id % 3};

    // Up
    return Move{(move_id - 32) / 4, move_id % 4, (move_id - 36) / 4, move_id % 4};

}

// Convert place move to int
int encode_place(int ptype, int row, int col) {

    return 48 + ptype * 16 + row * 4 + col;

}

// Convert move move to int
int encode_move(int row1, int col1, int row2, int col2) {

    // Right
    if (col1 < col2) return row1 * 3 + col1;

    // Down
    if (row1 < row2) return 12 + row1 * 4 + col1;

    // Left
    if (col1 > col2) return 24 + row1 * 3 + (col1 - 1);

    // Up
    return 36 + (row1 - 1) * 4 + col1;

}
