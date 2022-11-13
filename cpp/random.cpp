#include <bits/stdc++.h>
#include "game.h"
using namespace std;

class Node {
	      // Visits is number of times this node has been searched
          Game game;
	      unsigned short depth, edge;
	      float evaluation;
          short probabilities[56];
          unsigned int parent, visits;
          Node *reference;
	      //bitset<16> legal_moves;
};

int main() {
    vector<vector<Node>> x;
    for (int i = 0; i < 10000; ++i) {
        vector<Node> t;
        t.reserve(100000);
        x.push_back(t);
        cout << x.size() << ' ' << x.size() * 10000 * sizeof(Node) << ' ' << sizeof(Node) << ' ' << i << '\n';
    }
}