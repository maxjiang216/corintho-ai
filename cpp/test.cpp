#include <bits/stdc++.h>
#include "move.h"
#include "game.h"

using namespace std;

int main() {

    cout << "sizeof(Move): " << sizeof(Move) << '\n';
    cout << "sizeof(Game): " << sizeof(Game) << '\n';
    cout << "sizeof(bitset<48>): " << sizeof(bitset<48>) << '\n';
    cout << "sizeof(Node): " << sizeof(Node) << '\n';
    cout << bitset<96>(0xfffffffffffffffffffff);
    Game{}.get_legal_moves(bitset<96>{}.set());
    return 0;

}
