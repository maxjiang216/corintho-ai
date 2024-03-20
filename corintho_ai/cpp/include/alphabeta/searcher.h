#ifndef ALPHABETA_SEARCHER_H
#define ALPHABETA_SEARCHER_H

#include "alphabeta/node.h"

namespace AlphaBeta {

class Searcher {
public:
    Searcher(const Node& rootNode);
    void searchUpToDepth(int maxDepth);
    const Node& getRootNode() const;

private:
    Node rootNode;
};

} // namespace AlphaBeta

#endif // ALPHABETA_SEARCHER_H
