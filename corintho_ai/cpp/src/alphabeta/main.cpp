#include "alphabeta/node.h"
#include "alphabeta/dtypes.h"
#include "alphabeta/searcher.h"
#include <iostream>

int main(){
    AlphaBeta::Node node = AlphaBeta::Node();
    while (true){
        AlphaBeta::Searcher searcher = AlphaBeta::Searcher(node);
        searcher.searchUpToDepth(4);
        std::cout << searcher.getRootNode().getScore() << std::endl;
        std::cout << searcher.getRootNode().getGameState() << std::endl;
        std::cout << searcher.getRootNode().getBestMove().value() << std::endl;
        node = searcher.getRootNode().getChildren()[0];
        std::bitset<kNumMoves> legal_moves;
        node.getGameState().getLegalMoves(legal_moves);
        if (legal_moves.count() == 0){
            break;
        }
    }
    return 0;
}