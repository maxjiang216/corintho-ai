#include "game.h"
#include "move.h"
#include "util.h"
#include <vector>
#include <string>
#include <array>
#include <iostream>
using std::cout;

using std::vector;
using std::string;

// Used to represent vertical and horizontal lines in a position

// Is there a better way to do this?
// Arithmetic needs to be done on them
const uintf RL = 0;
const uintf RR = 1;
const uintf RB = 2;
const uintf CU = 3;
const uintf CD = 4;
const uintf CB = 5;

const uintf D0U = 0;
const uintf D0D = 1;
const uintf D0B = 2;
const uintf D1U = 3;
const uintf D1D = 4;
const uintf D1B = 5;
const uintf S0 = 6;
const uintf S1 = 7;
const uintf S2 = 8;
const uintf S3 = 9;

Game::Game(): board{bitset<3*BOARD_SIZE>().set()}, frozen{bitset<BOARD_SIZE>().set()}, to_play{0},
pieces{4, 4, 4, 4, 4, 4}, result{NONE} {}

bool Game::is_empty(uintf row, uintf col) {
    return !(
        board.test(row * 12 + col * 3 + 0) ||
	board.test(row * 12 + col * 3 + 1) ||
	board.test(row * 12 + col * 3 + 2)
    ); // Is there a faster way to test this given the consecutive addresses?
}

bool Game::can_place(uintf ptype, uintf row, uintf col) {

    // Check if space is empty
    // This is more common than frozen spaces
    // An empty space cannot be frozen
    if (is_empty(row, col)) return true;
    // Check if player has the piece left
    if (pieces[to_play * 3 + ptype]  == 0) return false;
    // Check if space is frozen
    if (frozen.test(row * 4 + col)) return false;
    // Bases can only be placed on empty spaces
    if (ptype == 0) return false; // is if (!ptype) faster?
    // Column
    // Check for absence of column and capital
    if (ptype == 1) {
        return !(
            board.test(row * 12 + col * 3 + 1) ||
	    board.test(row * 12 + col * 3 + 2)
	);
    }
    // Capital
    // Check for absence of base without column and capital
    return !(
        board.test(row * 12 + col * 3 + 2) ||
	(board.test(row * 12 + col * 3 + 0) && !board.test(row * 12 + col * 3 + 1))
    );
}

// Returns an int representing the bottom piece of a stack, 3 if empty
// Used to determine the legality of move moves
intf Game::get_bottom(uintf row, uintf col) {
    if (board.test(row * 12 + col * 3 + 0)) return 0;
    if (board.test(row * 12 + col * 3 + 1)) return 1;
    if (board.test(row * 12 + col * 3 + 2)) return 2;
    // Empty
    return 3;
}

// Returns an int representing the top of a stack, -1 if empty
// Used to determine the legality of move moves
intf Game::get_top(uintf row, uintf col) {
    if (board.test(row * 12 + col * 3 + 2)) return 2;
    if (board.test(row * 12 + col * 3 + 1)) return 1;
    if (board.test(row * 12 + col * 3 + 0)) return 0;
    // Empty
    return -1;
}

// Returns whether it is legal to move a stack between spaces
bool Game::can_move(uintf row1, uintf col1, uintf row2, uintf col2) {
    // Empty spaces, move moves not possible
    if (is_empty(row1, col1) || is_empty(row2, col2)) return false;
    // Frozen spaces
    if (frozen.test(row1 * 4 + col1) || frozen.test(row2 * 4 + col2)) return false;
    // The bottom of the first stack must go on the top of the second
    return get_bottom(row1, col1) - get_top(row2, col2) == 1;
}

// Returns whether a move is legal
bool Game::is_legal(uintf move_id) {
    Move move{move_id};
    // Place
    if (move.mtype) return can_place(move.ptype, move.row1, move.col1);
    // Move
    return can_move(move.row1, move.col1, move.row2, move.col2);    
}

// can be static (or just a normal function, although it is only used by game)
// we could also have it not be static and change a member that tracks whether there are lines
// since this is only called if there is a line
void apply_line(bitset<NUM_MOVES> &legal_moves, uintf line) {
    legal_moves &= line_breakers[line];
}

// Finds lines and moves that break all lines
// Returns whether there were lines
bool Game::get_line_breakers(bitset<NUM_MOVES> &legal_moves) {

    bool is_lines = false;
    intf space_0, space_1, space_2, space_3;

    // Rows
    for (uintf i = 0; i < 4; ++i) {
        space_1 = get_top(i, 1);
        if (space_1 != -1) { // If not empty
            space_2 = get_top(i, 2);
            if (space_1 == space_2) {
                space_0 = get_top(i, 0), space_3 = get_top(i, 3);
                if (space_0 == space_2) {
                    if (space_0 == space_3) {
                        is_lines = true;
                        apply_line(legal_moves, RB * 12 + i * 3 + space_0);
                    }
                    else {
                        is_lines = true;
                        apply_line(legal_moves, RL * 12 + i * 3 + space_0);
                    }
                }
                else if (space_2 == space_3) {
                    is_lines = true;
                    apply_line(legal_moves, RR * 12 + i * 3 + space_3);
                }
            }
        }
    }

    // Columns
    for (uintf j = 0; j < 4; ++j) {
        space_1 = get_top(1, j);
        if (space_1 != -1) { // If not empty
            space_2 = get_top(2, j);
	        if (space_1 == space_2) {
	            space_0 = get_top(0, j), space_3 = get_top(3, j);
	            if (space_0 == space_2) {
	                if (space_0 == space_3) {
                        is_lines = true;
		                apply_line(legal_moves, CB * 12 + j * 3 + space_0);
		            }
		            else {
                        is_lines = true;
		                apply_line(legal_moves, CU * 12 + j * 3 + space_0);
		            }
                }
	            else if (space_2 == space_3) {
                    is_lines = true;
	                apply_line(legal_moves, CD * 12 + j * 3 + space_3);
	            }
	        }
	    }
    }

    // Upper left to lower right long diagonal
    space_1 = get_top(1, 1);
    if (space_1 != -1) { // If not empty
        space_2 = get_top(2, 2);
	    if (space_1 == space_2) {
            space_0 = get_top(0, 0), space_3 = get_top(3, 3);
	        if (space_0 == space_2) {
	            if (space_0 == space_3) {
                    is_lines = true;
	                apply_line(legal_moves, 72 + D0B * 3 + space_0);
		        }
                else {
                    is_lines = true;
                    apply_line(legal_moves, 72 + D0U * 3 + space_0);
                }
	        }
            else if (space_2 == space_3) {
                is_lines = true;
                apply_line(legal_moves, 72 + D0D * 3 + space_3);
            }
	    }
    }

    // Upper right to lower left long diagonal
    space_1 = get_top(1, 2);
    if (space_1 != -1) { // If not empty
        space_2 = get_top(2, 1);
	    if (space_1 == space_2) {
	        space_0 = get_top(0, 3), space_3 = get_top(3, 0);
	        if (space_0 == space_2) {
	            if (space_0 == space_3) {
                    is_lines = true;
		            apply_line(legal_moves, 72 + D1B * 3 + space_0);
		        }
		        else {
                    is_lines = true;
                    apply_line(legal_moves, 72 + D1U * 3 + space_0);
		        }
	        }
	        else if (space_2 == space_3) {
                is_lines = true;
                apply_line(legal_moves, 72 + D1D * 3 + space_3);
	        }
	    }
    }

    // Top left short diagonal
    space_1 = get_top(1, 1);
    if (space_1 != -1 && space_1 == get_top(0, 2) && space_1 == get_top(2, 0)) {
        is_lines = true;
        apply_line(legal_moves, 72 + S0 * 3 + space_1);
    }

    // Top right short diagonal
    space_1 = get_top(1, 2);
    if (space_1 != -1 && space_1 == get_top(0, 1) && space_1 == get_top(2, 3)) {
        is_lines = true;
        apply_line(legal_moves, 72 + S1 * 3 + space_1);
    }

    // Bottom right short diagonal
    space_1 = get_top(2, 2);
    if (space_1 != -1 && space_1 == get_top(1, 3) && space_1 == get_top(3, 1)) {
        is_lines = true;
        apply_line(legal_moves, 72 + S2 * 3 + space_1);
    }

    // Bottom left short diagonal
    space_1 = get_top(2, 1);
    if (space_1 != -1 && space_1 == get_top(1, 0) && space_1 == get_top(3, 2)) {
        is_lines = true;
        apply_line(legal_moves, 72 + S3 * 3 + space_1);
    }

    return is_lines;

}

// Find lines and then filters through remaining moves to find legal moves
void Game::get_legal_moves(bitset<NUM_MOVES> &legal_moves) {

    // Set all bits to 1
    // Since apply_line does &=, we need to do this
    legal_moves.set();

    // Filter out moves that don't break lines
    bool is_lines = get_line_breakers(legal_moves);

    for (uintf i = 0; i < NUM_MOVES; ++i) {
        if (legal_moves.test(i) && !is_legal(i)) {
            legal_moves.flip(i);
	    }
    }

    // Terminal state
    if (legal_moves.none()) {
        // There is a line
        if (is_lines) {
            // Person to play is lost
            // Do we need to keep track of this? The current player is always the loser in this situation
            if (to_play == 0) {
                result = LOSS;
            }
            else {
                result = WIN;
            }
        }
        else {
            result = DRAW;
        }
    }

}

// Apply move to game. Does not check for legality
void Game::do_move(uintf move_id) {
    
    Move move{move_id};

    cout << "do_move 294 " << to_play << ' ' << move.ptype << '\n';

    // Reset which space is frozen
    frozen.reset();

    // Place
    if (move.mtype) {
        pieces[to_play * 3 + move.ptype] -= 1;
	    board.set(move.row1 * 12 + move.col1 * 3 + move.ptype);
    }
    // Move
    else {
        for (uintf i = 0; i < 3; ++i) {
	        board.set(
	            move.row2 * 12 + move.col2 * 3 + i,
		        board.test(move.row1 * 12 + move.col1 * 3 + i) ||
		        board.test(move.row2 * 12 + move.col2 * 3 + i)
	        );
            board.reset(move.row1 * 12 + move.col1 * 3 + i);
	    }
    }
    to_play = 1 - to_play;

    cout << "do_move 319 " << to_play << ' ' << move.ptype << '\n';

}

bool Game::is_terminal() {
    return result != NONE;
}

void Game::write_game_state(float game_state[GAME_STATE_SIZE]) {
    for (uintf i = 0; i < 3 * BOARD_SIZE; ++i) {
        if (board[i]) {
            game_state[i] = 1.0;
        }
        else {
            game_state[i] = 0.0;
        }
    }
    for (uintf i = 0; i < BOARD_SIZE; ++i) {
        if (frozen[i]) {
            game_state[3 * BOARD_SIZE + i] = 1.0;
        }
        else {
            game_state[3 * BOARD_SIZE + i] = 0.0;
        }
    }
    // Canonize the pieces
    for (uintf i = 0; i < 6; ++i) {
        game_state[4 * BOARD_SIZE + i] = (float)pieces[(to_play * 3 + i) % 6] * 0.25;
    }
}

// return result
Result Game::get_result() {
    return result;
}

uintf Game::get_to_play() {
    return to_play;
}