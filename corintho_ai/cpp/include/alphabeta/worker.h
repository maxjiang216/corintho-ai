#ifndef ALPHABETA_WORKER_H
#define ALPHABETA_WORKER_H

class Node; // Forward declaration

class Worker {
public:
    Worker(Node* node) : node_(node) {}

    void search();

private:
    Node* node_;
};

#endif // ALPHABETA_WORKER_H
