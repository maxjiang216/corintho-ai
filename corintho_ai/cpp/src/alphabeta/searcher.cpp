#include "alphabeta/searcher.h"

namespace AlphaBeta {

Searcher::Searcher(const Node& rootNode) : rootNode(rootNode) {
}

void Searcher::searchUpToDepth(int maxDepth) {
    Value alpha(Value::ValueType::NEGATIVE_INFINITY);
    Value beta(Value::ValueType::POSITIVE_INFINITY);
    for (int depth = 1; depth <= maxDepth; ++depth) {
        std::cout << "Depth: " << depth << std::endl;
        rootNode.search(depth, alpha, beta);
        rootNode.sortChildrenRecursively();
        for (const auto& child : rootNode.getChildren()) {
            std::cout << "Child:\n" << child.getGameState() << ", Score: " << child.getScore() << std::endl;
        }
    }
}

const Node& Searcher::getRootNode() const {
    return rootNode;
}

} // namespace AlphaBeta
