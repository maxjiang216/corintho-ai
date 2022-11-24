#ifndef NODE_H
#define NODE_H

#include "game.h"
#include "util.h"
#include <bitset>

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

    // Evaluations
    float evaluation, probabilities[NUM_MOVES];

    // Which children have been visited
    std::bitset<NUM_MOVES> visited;

  public:
    
    // Used to create root nodes
    Node();
    // Occasionally need to create root nodes from arbitrary game states
    Node(const Game &game);
    Node(const Game &game, uintf depth);
    // Used when writing into a new node
    // Will copy a game, then apply the move
    Node(const Game &game, uintf depth, uintf parent, uintf move_choice, uintf pos);
    ~Node() = default;

    // overwrite relevant parts of node
    void overwrite();
    void overwrite(const Game &new_game, uintf new_depth);
    void overwrite(const Game &new_game, uintf new_depth, uintf new_parent,
                   uintf move_choice, uintf pos, bool);

    // Accessors
    const Game& get_game() const;
    uintf get_to_play() const;
    bool is_legal(uintf id) const;
    Result get_result() const;
    bool is_terminal() const;
    uintf get_visits() const;
    uintf get_depth() const;
    uintf get_parent() const;
    float get_evaluation() const;
    bool has_visited(uintf move_choice) const;
    float get_probability(uintf move_choice) const;

    // Modifiers
    void increment_visits();
    void null_parent();
    void add_evaluation(float new_evalution);
    void set_probability(uintf move_choice, float probability);
    void adjust_probability(uintf move_choice, float scalar, float noise);
    void set_visit(uintf move_choice);

    void write_game_state(float game_state[GAME_STATE_SIZE]) const;

  };

#endif