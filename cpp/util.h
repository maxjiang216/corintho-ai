#ifndef UTIL_H
#define UTIL_H

#include <cstdint>
#include <string>
#include <bitset>
#include <array>

using std::bitset;
using std::string;

// Typedef

// Default ints to use
// Generally memory is not too critical, prioritize speed
typedef uint_fast32_t uintf;
typedef int_fast32_t intf;

// Ints for memory critical parts
// Generally use in large arrays
typedef uint_least8_t uint8s;
typedef uint_least16_t uint16s;

// Define constants used by the pipeline
// This is for constants used by more than one file
// Otherwise keep the constants in the file that uses it

// Size of game state sample
const uintf GAME_STATE_SIZE = 70;
// Number of possible legal moves
const uintf NUM_TOTAL_MOVES = 96;
// Number of moves the simulator uses
const uintf NUM_MOVES = 96;

enum Result {NONE, LOSS, DRAW, WIN};

inline std::array<bitset<NUM_TOTAL_MOVES>, 102> line_breakers = {
    bitset<NUM_TOTAL_MOVES>(string("001000000000111000000000000000000000111000000000000100000000000011100000000000000000000000000000")),
    bitset<NUM_TOTAL_MOVES>(string("001000000000111000000000000000000000111000000000000000000000000000010000000000001110000000000000")),
    bitset<NUM_TOTAL_MOVES>(string("001000000000111000000000000000000000111000000000000000000000000000000000000000000001000000000000")),
    bitset<NUM_TOTAL_MOVES>(string("000001000000111011100000000000000000111011100000000000010000000000001110000000000000000000000000")),
    bitset<NUM_TOTAL_MOVES>(string("000001000000111011100000000000000000111011100000000000000000000000000001000000000000111000000000")),
    bitset<NUM_TOTAL_MOVES>(string("000001000000111011100000000000000000111011100000000000000000000000000000000000000000000100000000")),
    bitset<NUM_TOTAL_MOVES>(string("000000001000000011101110000000000000000011101110000000000001000000000000111000000000000000000000")),
    bitset<NUM_TOTAL_MOVES>(string("000000001000000011101110000000000000000011101110000000000000000000000000000100000000000011100000")),
    bitset<NUM_TOTAL_MOVES>(string("000000001000000011101110000000000000000011101110000000000000000000000000000000000000000000010000")),
    bitset<NUM_TOTAL_MOVES>(string("000000000001000000001110000000000000000000001110000000000000000100000000000011100000000000000000")),
    bitset<NUM_TOTAL_MOVES>(string("000000000001000000001110000000000000000000001110000000000000000000000000000000010000000000001110")),
    bitset<NUM_TOTAL_MOVES>(string("000000000001000000001110000000000000000000001110000000000000000000000000000000000000000000000001")),
    bitset<NUM_TOTAL_MOVES>(string("000000000000011100000000100000000000011100000000100000000000000001110000000000000000000000000000")),
    bitset<NUM_TOTAL_MOVES>(string("000000000000011100000000100000000000011100000000000000000000000010000000000000000111000000000000")),
    bitset<NUM_TOTAL_MOVES>(string("000000000000011100000000100000000000011100000000000000000000000000000000000000001000000000000000")),
    bitset<NUM_TOTAL_MOVES>(string("000000000000011101110000000100000000011101110000000010000000000000000111000000000000000000000000")),
    bitset<NUM_TOTAL_MOVES>(string("000000000000011101110000000100000000011101110000000000000000000000001000000000000000011100000000")),
    bitset<NUM_TOTAL_MOVES>(string("000000000000011101110000000100000000011101110000000000000000000000000000000000000000100000000000")),
    bitset<NUM_TOTAL_MOVES>(string("000000000000000001110111000000100000000001110111000000001000000000000000011100000000000000000000")),
    bitset<NUM_TOTAL_MOVES>(string("000000000000000001110111000000100000000001110111000000000000000000000000100000000000000001110000")),
    bitset<NUM_TOTAL_MOVES>(string("000000000000000001110111000000100000000001110111000000000000000000000000000000000000000010000000")),
    bitset<NUM_TOTAL_MOVES>(string("000000000000000000000111000000000100000000000111000000000000100000000000000001110000000000000000")),
    bitset<NUM_TOTAL_MOVES>(string("000000000000000000000111000000000100000000000111000000000000000000000000000010000000000000000111")),
    bitset<NUM_TOTAL_MOVES>(string("000000000000000000000111000000000100000000000111000000000000000000000000000000000000000000001000")),
    bitset<NUM_TOTAL_MOVES>(string("000000000000011000000000000000000000011000000000000000000000000001100000000000000000000000000000")),
    bitset<NUM_TOTAL_MOVES>(string("000000000000011000000000000000000000011000000000000000000000000000000000000000000110000000000000")),
    bitset<NUM_TOTAL_MOVES>(string("000000000000011000000000000000000000011000000000000000000000000000000000000000000000000000000000")),
    bitset<NUM_TOTAL_MOVES>(string("000000000000011001100000000000000000011001100000000000000000000000000110000000000000000000000000")),
    bitset<NUM_TOTAL_MOVES>(string("000000000000011001100000000000000000011001100000000000000000000000000000000000000000011000000000")),
    bitset<NUM_TOTAL_MOVES>(string("000000000000011001100000000000000000011001100000000000000000000000000000000000000000000000000000")),
    bitset<NUM_TOTAL_MOVES>(string("000000000000000001100110000000000000000001100110000000000000000000000000011000000000000000000000")),
    bitset<NUM_TOTAL_MOVES>(string("000000000000000001100110000000000000000001100110000000000000000000000000000000000000000001100000")),
    bitset<NUM_TOTAL_MOVES>(string("000000000000000001100110000000000000000001100110000000000000000000000000000000000000000000000000")),
    bitset<NUM_TOTAL_MOVES>(string("000000000000000000000110000000000000000000000110000000000000000000000000000001100000000000000000")),
    bitset<NUM_TOTAL_MOVES>(string("000000000000000000000110000000000000000000000110000000000000000000000000000000000000000000000110")),
    bitset<NUM_TOTAL_MOVES>(string("000000000000000000000110000000000000000000000110000000000000000000000000000000000000000000000000")),
    bitset<NUM_TOTAL_MOVES>(string("100100100000000000001000100100100000000000000000000000000000100010001000100000000000000000000000")),
    bitset<NUM_TOTAL_MOVES>(string("100100100000000000001000100100100000000000000000000000000000000000000000000010001000100010000000")),
    bitset<NUM_TOTAL_MOVES>(string("100100100000000000001000100100100000000000000000000000000000000000000000000000000000000000001000")),
    bitset<NUM_TOTAL_MOVES>(string("110110110000000000000100110110110000000000000000000000000000010001000100010000000000000000000000")),
    bitset<NUM_TOTAL_MOVES>(string("110110110000000000000100110110110000000000000000000000000000000000000000000001000100010001000000")),
    bitset<NUM_TOTAL_MOVES>(string("110110110000000000000100110110110000000000000000000000000000000000000000000000000000000000000100")),
    bitset<NUM_TOTAL_MOVES>(string("011011011000000000000010011011011000000000000000000000000000001000100010001000000000000000000000")),
    bitset<NUM_TOTAL_MOVES>(string("011011011000000000000010011011011000000000000000000000000000000000000000000000100010001000100000")),
    bitset<NUM_TOTAL_MOVES>(string("011011011000000000000010011011011000000000000000000000000000000000000000000000000000000000000010")),
    bitset<NUM_TOTAL_MOVES>(string("001001001000000000000001001001001000000000000000000000000000000100010001000100000000000000000000")),
    bitset<NUM_TOTAL_MOVES>(string("001001001000000000000001001001001000000000000000000000000000000000000000000000010001000100010000")),
    bitset<NUM_TOTAL_MOVES>(string("001001001000000000000001001001001000000000000000000000000000000000000000000000000000000000000001")),
    bitset<NUM_TOTAL_MOVES>(string("000100100100000000000000000100100100100000000000100000000000000000001000100010000000000000000000")),
    bitset<NUM_TOTAL_MOVES>(string("000100100100000000000000000100100100100000000000000000000000000010000000000000000000100010001000")),
    bitset<NUM_TOTAL_MOVES>(string("000100100100000000000000000100100100100000000000000000000000000000000000000000001000000000000000")),
    bitset<NUM_TOTAL_MOVES>(string("000110110110000000000000000110110110010000000000010000000000000000000100010001000000000000000000")),
    bitset<NUM_TOTAL_MOVES>(string("000110110110000000000000000110110110010000000000000000000000000001000000000000000000010001000100")),
    bitset<NUM_TOTAL_MOVES>(string("000110110110000000000000000110110110010000000000000000000000000000000000000000000100000000000000")),
    bitset<NUM_TOTAL_MOVES>(string("000011011011000000000000000011011011001000000000001000000000000000000010001000100000000000000000")),
    bitset<NUM_TOTAL_MOVES>(string("000011011011000000000000000011011011001000000000000000000000000000100000000000000000001000100010")),
    bitset<NUM_TOTAL_MOVES>(string("000011011011000000000000000011011011001000000000000000000000000000000000000000000010000000000000")),
    bitset<NUM_TOTAL_MOVES>(string("000001001001000000000000000001001001000100000000000100000000000000000001000100010000000000000000")),
    bitset<NUM_TOTAL_MOVES>(string("000001001001000000000000000001001001000100000000000000000000000000010000000000000000000100010001")),
    bitset<NUM_TOTAL_MOVES>(string("000001001001000000000000000001001001000100000000000000000000000000000000000000000001000000000000")),
    bitset<NUM_TOTAL_MOVES>(string("000100100000000000000000000100100000000000000000000000000000000000001000100000000000000000000000")),
    bitset<NUM_TOTAL_MOVES>(string("000100100000000000000000000100100000000000000000000000000000000000000000000000000000100010000000")),
    bitset<NUM_TOTAL_MOVES>(string("000100100000000000000000000100100000000000000000000000000000000000000000000000000000000000000000")),
    bitset<NUM_TOTAL_MOVES>(string("000110110000000000000000000110110000000000000000000000000000000000000100010000000000000000000000")),
    bitset<NUM_TOTAL_MOVES>(string("000110110000000000000000000110110000000000000000000000000000000000000000000000000000010001000000")),
    bitset<NUM_TOTAL_MOVES>(string("000110110000000000000000000110110000000000000000000000000000000000000000000000000000000000000000")),
    bitset<NUM_TOTAL_MOVES>(string("000011011000000000000000000011011000000000000000000000000000000000000010001000000000000000000000")),
    bitset<NUM_TOTAL_MOVES>(string("000011011000000000000000000011011000000000000000000000000000000000000000000000000000001000100000")),
    bitset<NUM_TOTAL_MOVES>(string("000011011000000000000000000011011000000000000000000000000000000000000000000000000000000000000000")),
    bitset<NUM_TOTAL_MOVES>(string("000001001000000000000000000001001000000000000000000000000000000000000001000100000000000000000000")),
    bitset<NUM_TOTAL_MOVES>(string("000001001000000000000000000001001000000000000000000000000000000000000000000000000000000100010000")),
    bitset<NUM_TOTAL_MOVES>(string("000001001000000000000000000001001000000000000000000000000000000000000000000000000000000000000000")),
    bitset<NUM_TOTAL_MOVES>(string("000110011000011001000010000110011000010001100010000000000000000000000100001000000000000000000000")),
    bitset<NUM_TOTAL_MOVES>(string("000110011000011001000010000110011000010001100010000000000000000000000000000000000000010000100000")),
    bitset<NUM_TOTAL_MOVES>(string("000110011000011001000010000110011000010001100010000000000000000000000000000000000000000000000000")),
    bitset<NUM_TOTAL_MOVES>(string("000110011000011001000010000110011000010001100010000000000000000000000100001000000000000000000000")),
    bitset<NUM_TOTAL_MOVES>(string("000110011000011001000010000110011000010001100010000000000000000000000000000000000000010000100000")),
    bitset<NUM_TOTAL_MOVES>(string("000110011000011001000010000110011000010001100010000000000000000000000000000000000000000000000000")),
    bitset<NUM_TOTAL_MOVES>(string("000110011000011001000010000110011000010001100010000000000000000000000100001000000000000000000000")),
    bitset<NUM_TOTAL_MOVES>(string("000110011000011001000010000110011000010001100010000000000000000000000000000000000000010000100000")),
    bitset<NUM_TOTAL_MOVES>(string("000110011000011001000010000110011000010001100010000000000000000000000000000000000000000000000000")),
    bitset<NUM_TOTAL_MOVES>(string("000011110000011000100100000011110000001001100100000000000000000000000010010000000000000000000000")),
    bitset<NUM_TOTAL_MOVES>(string("000011110000011000100100000011110000001001100100000000000000000000000000000000000000001001000000")),
    bitset<NUM_TOTAL_MOVES>(string("000011110000011000100100000011110000001001100100000000000000000000000000000000000000000000000000")),
    bitset<NUM_TOTAL_MOVES>(string("000011110000011000100100000011110000001001100100000000000000000000000010010000000000000000000000")),
    bitset<NUM_TOTAL_MOVES>(string("000011110000011000100100000011110000001001100100000000000000000000000000000000000000001001000000")),
    bitset<NUM_TOTAL_MOVES>(string("000011110000011000100100000011110000001001100100000000000000000000000000000000000000000000000000")),
    bitset<NUM_TOTAL_MOVES>(string("000011110000011000100100000011110000001001100100000000000000000000000010010000000000000000000000")),
    bitset<NUM_TOTAL_MOVES>(string("000011110000011000100100000011110000001001100100000000000000000000000000000000000000001001000000")),
    bitset<NUM_TOTAL_MOVES>(string("000011110000011000100100000011110000001001100100000000000000000000000000000000000000000000000000")),
    bitset<NUM_TOTAL_MOVES>(string("011110100000011011001000011110100000011011001000000000000000000000100100100000000000000000000000")),
    bitset<NUM_TOTAL_MOVES>(string("011110100000011011001000011110100000011011001000000000000000000000000000000000000010010010000000")),
    bitset<NUM_TOTAL_MOVES>(string("011110100000011011001000011110100000011011001000000000000000000000000000000000000000000000000000")),
    bitset<NUM_TOTAL_MOVES>(string("110011001000011000110001110011001000011000110001000000000000000001000010000100000000000000000000")),
    bitset<NUM_TOTAL_MOVES>(string("110011001000011000110001110011001000011000110001000000000000000000000000000000000100001000010000")),
    bitset<NUM_TOTAL_MOVES>(string("110011001000011000110001110011001000011000110001000000000000000000000000000000000000000000000000")),
    bitset<NUM_TOTAL_MOVES>(string("000001011110000100110110000001011110000100110110000000000000000000000001001001000000000000000000")),
    bitset<NUM_TOTAL_MOVES>(string("000001011110000100110110000001011110000100110110000000000000000000000000000000000000000100100100")),
    bitset<NUM_TOTAL_MOVES>(string("000001011110000100110110000001011110000100110110000000000000000000000000000000000000000000000000")),
    bitset<NUM_TOTAL_MOVES>(string("000100110011100011000110000100110011100011000110000000000000000000001000010000100000000000000000")),
    bitset<NUM_TOTAL_MOVES>(string("000100110011100011000110000100110011100011000110000000000000000000000000000000000000100001000010")),
    bitset<NUM_TOTAL_MOVES>(string("000100110011100011000110000100110011100011000110000000000000000000000000000000000000000000000000"))
};

#endif