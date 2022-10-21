#include "trainmc.h"
#include "game.h"
#include "move.h"
#include <memory>
#include <bitset>

using std::unique_ptr;
using std::bitset;

// Only used to create the root node (starting position)
TrainMC::Node::Node(): game{Game()}, visits{0}, depth{0}, evaluation{0}, legal_moves{unique_ptr<bitset<96>>()->set()}, parent{nullptr} {}

TrainMC::Node::Node(Game game, int depth, Node *parent): game{game}, visits{0}, depth{depth}, parent{parent} {}

// Used to initialize tree with root node
TrainMC::TrainMC(): root{shared_ptr(new Node{})}, iterations_done{0}, cur_node{nullptr} {}


