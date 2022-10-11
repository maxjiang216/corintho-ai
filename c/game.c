#include "game.h"

// Initialize a game object to the starting position
struct Game* init_game() {

    struct Game *game = (struct Game *) malloc(sizeof(struct Game));

    for (size_t i = 0; i < 48; ++i) game->board[i] = false;
    for (size_t i = 0; i < 16; ++i) game->frozen[i] = true;
    game->frozen[0 * 4 + 0] = false;
    game->frozen[0 * 4 + 1] = false;
    game->frozen[1 * 4 + 1] = false;
    game->to_play = 0;
    for (size_t i = 0; i < 6; ++i) game->pieces = 4;
    game->is_done = false;
    // We don't need to initialize game->outcome until the game is complete
}

// Returns whether space is empty
bool is_empty(struct Game* game, short row, short col) {
    return (
        game->board[row * 12 + col * 3 + 0] ||
	game->board[row * 12 + col * 3 + 1] ||
	game->board[row * 12 + col * 3 + 2]
    );
}

// Returns whether piece can be placed at space
bool can_place(struct Game* game, short ptype, short row, short col) {
    
    // Check if space is frozen
    // Empty spaces should be more common, should we check that first?
    if (game->frozen[row * 4 + col]) return false;
    // Check if space is empty
    if (is_empty(game, row, col)) return true;
    // Base
    if (ptype == 0) return false;
    // Column
    // Check fror absence of column and capital
    if (ptype == 1) {
        return (
            !game->board[row * 12 + col * 3 + 1] &&
	    !game->board[row * 12 + col * 3 + 2]
	);
    }
    // Capital
    // Check for absence of base and capital
    return (
        !game->board[row * 12 + col * 3 + 0] &&
	!game->board[row * 12 + col * 3 + 2]
    );

}

// Returns the bottom of a stack, 3 if empty
short get_bottom(struct Game *game, short row, short col) {
    if (game->board[row * 12 + col * 3 + 0]) return 0;
    if (game->board[row * 12 + col * 3 + 1]) return 1;
    if (game->board[row * 12 + col * 3 + 2]) return 2;
    // Empty
    return 3;
}

// Returns the top of a stack, -1 if empty
short get_top(struct Game *game, short row, short col) {
    if (game->board[row * 12 + col * 3 + 2]) return 2;
    if (game->board[row * 12 + col * 3 + 1]) return 1;
    if (game->board[row * 12 + col * 3 + 0]) return 0;
    // Empty
    return -1;
}
// Returns whether it is legal to move between spaces
bool can_move(struct Game* game, short row1, short col1, short row2, short col2) {

    // Empty spaces
    if (is_empty(game, row1, col1) || is_empty(game, row2, col2)) return false;
    // Frozen spaces
    if (game->frozen[row1 * 4 + col1] || game->frozen[row2 * 4 + col2]) return false;
    return get_bottom(game, row1, col1) - get_top(game, row2, col2) == 1;

}

// Returns whether a move is legal
bool is_legal(struct Game*, short move_id) {
    
    struct Move move = get_move(move_id);
    // Place
    if (move->mtype) return can_place(move->ptype, move->row1, move->col1);
    // Move
    return can_move(move->row1, move->col1, move->row2, move->col2);

}

bool* get_legal_moves(struct Game* game) {
 // 
}

struct Do_Move_Output do_move(struct Game*, short) {


}

double* get_vector(struct Game*);

#endif
