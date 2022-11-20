#include <stdlib.h>
#include "move.h"

// Convert move_id to Move object
// We can make this a table in the future, especially if we incude rotations (?)
Move::Move(uintf move_id) {

    // Place
    // I believe that place is more common
    if (move_id >= 48) {
        ptype = (move_id - 48) / 16;
        row1 = (move_id % 16) / 4;
        col1 = move_id % 4;
    }

    // Move

    // Right
    else if (move_id < 12) {
        row1 = move_id / 3;
        col1 = move_id % 3;
        row2 = move_id / 3;
        col2 = move_id % 3 + 1;
    }
	
    // Down
    else if (move_id < 24) {
        row1 = (move_id - 12) / 4;
        col1 = move_id % 4;
        row2 = (move_id - 8) / 4;
        col2 = move_id % 4;
    }

    // Left
    else if (move_id < 36) {
        row1 = (move_id - 24) / 3;
        col1 = move_id % 3 + 1;
        row2 = (move_id - 24) / 3;
        col2 = move_id % 3;
    }

    // Up
    else {
        row1 = (move_id - 32) / 4;
        col1 = move_id % 4;
        row2 = (move_id - 36) / 4;
        col2 = move_id % 4;
    }
	
}

// Convert place move to int
uintf encode_place(uintf ptype, uintf row, uintf col) {

    return 48 + ptype * 16 + row * 4 + col;

}

// Convert move move to int
uintf encode_move(uintf row1, uintf col1, uintf row2, uintf col2) {

    // Right
    if (col1 < col2) return row1 * 3 + col1;

    // Down
    if (row1 < row2) return 12 + row1 * 4 + col1;

    // Left
    if (col1 > col2) return 24 + row1 * 3 + (col1 - 1);

    // Up
    return 36 + (row1 - 1) * 4 + col1;

}