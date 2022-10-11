#include <stdlib.h>
#include "move.h"

struct Move move_from_id(short move_id) {

    // Place
    // I believe that place is more common
    if (move_id >= 48) {
	return (struct Move){
	    .mtype = true,
            .ptype = (move_id - 48) / 16,
	    .row1 = (move_id % 16) / 4,
	    .col1 = move_id % 4
        };
    }

    // Move

    // Right
    if (move_id < 12) {
        return (struct Move){
            .mtype = false,
	    .row1 = move_id / 3,
	    .col1 = move_id % 3,
	    .row2 = move_id / 3,
	    .col2 = move_id % 3 + 1
        };
    }

    // Down
    if (move_id < 24) {
	return (struct Move){
	    .mtype = false,
	    .row1 = (move_id - 12) / 4,
	    .col1 = move_id % 4,
	    .row2 = (move_id - 8) / 4,
	    .col2 = move_id % 4
        };
    }

    // Left
    if (move_id < 36) {
	return (struct Move){
	    .mtype = false,
            .row1 = (move_id - 24) / 3,
	    .col1 = move_id % 3 + 1,
	    .row2 = (move_id - 24) / 3,
	    .col2 = move_id % 3
        };
    }

    // Up
    return (struct Move){
        .mtype = false,
        .row1 = (move_id - 32) / 4,
	.col1 = move_id % 4,
	.row2 = (move_id - 36) / 4,
	.col2 = move_id % 4
    };

}

short encode_place(short ptype, short row, short col) {

    return 48 + ptype * 16 + row * 4 + col;

}

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
