#include "alphabeta/node.h"
#include "alphabeta/dtypes.h"
#include <iostream>

int main(){
    AlphaBeta::Node node = AlphaBeta::Node();
    node.search(4, AlphaBeta::Value(AlphaBeta::Value::ValueType::NEGATIVE_INFINITY), AlphaBeta::Value(AlphaBeta::Value::ValueType::POSITIVE_INFINITY));
    std::cout << node.score << std::endl;
    std::cout << node.gameState << std::endl;
    return 0;
}