#ifndef GAME_H
#define GAME_H
#include "move.h"

struct Game {

    bool board[48], frozen[16], is_done;
    short to_play, pieces[6], outcome;

};

struct Do_Move_Output {
    
    struct Game *game;
    bool legal_moves[96];

}

struct Game* init_game();
bool is_empty(struct Game*, short, short);
bool can_place(struct Game*, short, short, short);
bool can_move(struct Game*, short, short, short, short);
bool is_legal(struct Game*, short);
struct Do_Move_Output do_move(struct Game*, short);
bool* get_legal_moves(struct Game*);
double* get_vector(struct Game*);

#endif
