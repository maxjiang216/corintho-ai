#ifndef GAME_H
#define GAME_H

// Bitset can save on memory and make resetting all bits faster
#include <bitset>
#include <memory>

using std::bitset;
using std::shared_ptr;

class Game {

    bitset<48> board;
    bitset<16> frozen;
    int pieces[6], outcome, to_play;

    bool is_empty(int, int);
    bool can_place(int, int, int);
    int get_bottom(int, int);
    int get_top(int, int);
    bool can_move(int, int, int, int);
    bool is_legal(int);
    void get_line_breakers(bitset<96> &legal_moves);

  public:

    Game();
    void get_legal_moves(bitset<96> &legal_moves);

};

class Node {
    Game game;
    int searches, depth;
    Node *parent;
    bitset<96> legal_moves;
    float noisy_probabilities[96];
    float evaluations[96];
    shared_ptr<Node> children[96];
};
#endif
