#ifndef UTIL_H
#define UTIL_H

#include <cstdint>

// Typedef

// Default ints to use
// Generally memory is not too critical, prioritize speed
typedef int_fast8_t int8;
typedef uint_fast8_t uint8;
typedef uint_fast16_t uint16;
typedef uint_fast32_t uint32;

// Ints for memory critical parts
// Generally use in large arrays
typedef uint_least8_t uint8s;
typedef uint_least16_t uint16s;


// Define constants used by the pipeline
// This is for constants used by more than one file
// Otherwise keep the constants in the file that uses it

// Size of game state sample
const uint8 GAME_STATE_SIZE = 96;
// Number of possible legal moves
const uint8 NUM_TOTAL_MOVES = 96;
// Number of moves the simulator uses
const uint8 NUM_MOVES = 56;


#endif