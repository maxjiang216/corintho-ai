#include "foobar.h"

Bar::Bar(int x): f{Foo{x}} {}

int Bar::fun(int y) {
  return f.fun(y)+y;
}
