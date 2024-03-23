#include "alphabeta/searcher.h"

namespace AlphaBeta {

Searcher::Searcher(const Node& rootNode) : rootNode(rootNode) {
}

void Searcher::searchUpToDepth(int maxDepth) {
    Value alpha(Value::ValueType::NEGATIVE_INFINITY);
    Value beta(Value::ValueType::POSITIVE_INFINITY);
    for (int depth = 1; depth <= maxDepth; ++depth) {
        rootNode.search(depth, alpha, beta);
        rootNode.sortChildrenRecursively();
    }
}

const Node& Searcher::getRootNode() const {
    return rootNode;
}

} // namespace AlphaBeta
