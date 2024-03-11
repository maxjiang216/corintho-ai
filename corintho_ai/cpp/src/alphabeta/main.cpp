#include "alphabeta/node.h"
#include "alphabeta/dtypes.h"
#include <iostream>

int main(){
    AlphaBeta::Node node = AlphaBeta::Node();
    std::cout << node.score << std::endl;
    return 0;
}