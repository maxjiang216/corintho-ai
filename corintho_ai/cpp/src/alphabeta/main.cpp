#include "alphabeta/node.h"
#include "alphabeta/dtypes.h"
#include "alphabeta/searcher.h"
#include <iostream>

int main(){
    AlphaBeta::Node node = AlphaBeta::Node();
    AlphaBeta::Searcher searcher = AlphaBeta::Searcher(node);
    searcher.searchUpToDepth(4);
    std::cout << searcher.getRootNode().getScore() << std::endl;
    std::cout << searcher.getRootNode().getGameState() << std::endl;
    return 0;
}