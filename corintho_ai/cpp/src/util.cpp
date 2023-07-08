#include "util.h"

#include <string>

std::string strResult(Result result) {
  if (result == kResultLoss) {
    return "L";
  }
  if (result == kResultDraw) {
    return "D";
  }
  if (result == kResultWin) {
    return "W";
  }
  if (result == kDeducedLoss) {
    return "DL";
  }
  if (result == kDeducedDraw) {
    return "DD";
  }
  if (result == kDeducedWin) {
    return "DW";
  }
  return "N";
}