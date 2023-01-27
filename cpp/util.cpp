#include "util.h"
#include <iostream>

using std::cerr;

void print_line(uintf line) {
  if (line < 12) {
    cerr << "RL" << (line % 12) / 3;
  } else if (line < 24) {
    cerr << "RR" << (line % 12) / 3;
  } else if (line < 36) {
    cerr << "RB" << (line % 12) / 3;
  } else if (line < 48) {
    cerr << "CU" << (line % 12) / 3;
  } else if (line < 60) {
    cerr << "CD" << (line % 12) / 3;
  } else if (line < 72) {
    cerr << "CB" << (line % 12) / 3;
  } else if (line < 75) {
    cerr << "D0U";
  } else if (line < 78) {
    cerr << "D0D";
  } else if (line < 81) {
    cerr << "D0B";
  } else if (line < 84) {
    cerr << "D1U";
  } else if (line < 87) {
    cerr << "D1D";
  } else if (line < 90) {
    cerr << "D1B";
  } else if (line < 93) {
    cerr << "S0";
  } else if (line < 96) {
    cerr << "S1";
  } else if (line < 99) {
    cerr << "S2";
  } else {
    cerr << "S3";
  }

  switch (line % 3) {
  case 0:
    cerr << 'B';
    break;
  case 1:
    cerr << 'C';
    break;
  default:
    cerr << 'A';
  }
}

string str_result(uint8s result) {
  if (result == RESULT_LOSS) {
    return "L";
  }
  if (result == RESULT_DRAW) {
    return "D";
  }
  if (result == RESULT_WIN) {
    return "W";
  }
  if (result == DEDUCED_LOSS) {
    return "DL";
  }
  if (result == DEDUCED_DRAW) {
    return "DD";
  }
  if (result == DEDUCED_WIN) {
    return "DW";
  }
  return "N";
}