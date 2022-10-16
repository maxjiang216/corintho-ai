#include "game.h"
#include "move.h"
#include <vector.h>

using std::vector;

// Used to represent lines in a position
struct Line {
    int_least8_t direction, alignment, coordinate, ptype;
};

Game::Game(): board{bitset<48>{}}, frozen{bitset<48>{}}, is_done{false}, to_play{0}, pieces{4, 4, 4, 4, 4, 4} {}

bool Game::is_empty(int_fast8_t row, int_fast8_t col) {
    return !(
        board.test(row * 12 + col * 3 + 0) ||
	board.test(row * 12 + col * 3 + 1) ||
	board.test(row * 12 + col * 3  2)
    ); // Is there a faster way to test this given the consecutive addresses?
}

bool Game::can_place(uint_fast8_t ptype, uint_fast8_t row, uint_fast8_t col) {

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
uint_fast8_t Game::get_bottom(uint_fast8_t row, uint_fast8_t col) {
    if (board.test(row * 12 + col * 3 + 0)) return 0;
    if (board.test(row * 12 + col * 3 + 1)) return 1;
    if (board.test(row * 12 + col * 3 + 2)) return 2;
    // Empty
    return 3;
}

// Returns an int representing the top of a stack, -1 if empty
// Used to determine the legality of move moves
uint_fast8_t Game::get_top(uint_fast8_t row, uint_fast8_t col) {
    if (board.test(row * 12 + col * 3 + 2)) return 2;
    if (board.test(row * 12 + col * 3 + 1)) return 1;
    if (board.test(row * 12 + col * 3 + 0)) return 0;
    // Empty
    return -1;
}

// Returns whether it is legal to move a stack between spaces
bool Game::can_move(uint_fast8_t row1, uint_fast8_t col1, uint_fast8_t row2, uint_fast8_t col2) {
    // Empty spaces, move moves not possible
    if (is_empty(row1, col1) || is_empty(row2, col2)) return false;
    // Frozen spaces
    if (frozen.test(row1 * 4 + col1) || frozen(row2 * 4 + col2)) return false;
    // The bottom of the first stack must go on the top of the second
    return get_bottom(row1, col1) - get_top(row2, col2) == 1;
}

// Returns whether a move is legal
bool Game::is_legal(uint_fast8_t move_id) {
    Move move{move_id};
    // Place
    if (move.mtype) return can_place(move.ptype, move.row1, move.col1);
    // Move
    return can_move(move.row1, move.col1, move.row2, move.col2);    
}

// Get all moves that break the given line
static bitset<96> Game::get_moves_break_line(Line line) {

    bitset<96> moves{};

    if (line.direction == 0) { // Row
        vector<int_least8_t> sources;
	switch (line.coordinate) {
            case 0:
	        sources = vector<int_least8_t>{1};
		break;
	    case 1:
		sources = vector<int_least8_t>{0, 2};
		break;
	    case 2:
		sources = vector<int_least8_t>{1, 3};
	    default: // 2
		sources = 
	}
	if (line.coordinate == 0) {
            sources = vector<int_least8_t>{1};
	}
	else if (line.coordinate
    }
}

// Finds lines and moves that break all lines
bitset<96> get_moves_break_lines {

}
// Find lines and then filters through remaining moves to find legal moves
bitset<96> get_legal_moves() {

}
