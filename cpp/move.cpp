#include <stdlib.h>
#include "move.h"

// Nullary constructor
// Ideally not used
Move::Move() {}

// Convert move_id to Move object
Move get_move_from_id(short move_id) {

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

// 3 int constructor
// Used for place moves
Move::Move(short ptype, short row, short col): mtype{true}, ptype{ptype}, row1{row1}, col1{col} {}

// 4 int constructor
// Used for move moves
Move::Move(short row1, short col1, short row2, short co2): mtype{false}, row1{row1}, col1{col1}, row2{row2}, col2{col2} {}

// Convert place move to int
short encode_place(short ptype, short row, short col) {

    return 48 + ptype * 16 + row * 4 + col;

}

// Convert move move to int
short encode_move(short row1, short col1, short row2, short col2) {

    // Right
    if (col1 < col2) return row1 * 3 + col1;

    // Down
    if (row1 < row2) return 12 + row1 * 4 + col1;

    // Left
    if (col1 > col2) return 24 + row1 * 3 + (col1 - 1);

    // Up
    return 36 + (row1 - 1) * 4 + col1;

}
