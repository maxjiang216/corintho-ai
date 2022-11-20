#ifndef NODE_H
#define NODE_H

#include "game.h"
#include "util.h"
#include <bitset>
#include <array>

using std::bitset;

// Node in Monte Carlo Tree
  class Node {

    // Game state
    Game game;
    // visits is number of times this node has been searched
    uintf visits;
    // depth starts at 0
    uintf depth;
    // Index of parent node
    uintf parent;
    float evaluation;
    std::array<float, NUM_MOVES> probabilities;
    // legal_moves denotes which moves are legal, stores in node instead of game since Node is the one using it (?)
    // visited denotes which children have been visited, useful for many computations
    bitset<NUM_MOVES> legal_moves, visited;

  public:
    
    // Used to create the root node
    Node();
    Node(const Game &game);
    Node(const Game &game, uintf depth);
    // Used when writing into a new node
    // Will copy a game, then apply the move
    Node(const Game &game, uintf depth, uintf parent, uintf move_choice);
    ~Node() = default;

    // overwrite relevant parts of node
    // other things will be overwritten lazily
    void overwrite();
    void overwrite(const Game &new_game, uintf new_depth);
    void overwrite(const Game &new_game, uintf new_depth, uintf new_parent, uintf move_choice);

    bool is_terminal();
    uintf get_depth();
    const Game& get_game();
    Result get_result();
    bool has_visited(uintf move_choice);
    uintf get_visits();
    void add_evaluation(float new_evalution);
    float get_evaluation();
    uintf get_parent();
    void increment_visits();
    uintf get_to_play();
    void set_probability(uintf id, float probability);
    float get_probability(uintf id);
    void adjust_probability(uintf id, float scalar, float noise);
    bool is_legal(uintf id);
    void set_visit(uintf move_choice);

    void write_game_state(float game_state[GAME_STATE_SIZE]);

  };

#endif