#include "foobar.h"

Foo::Foo(int x): x{x} {}

int Foo::fun(int y) {
  return x+y;
}

Bar::Bar(int x): f{Foo{x}} {}

int Bar::fun(int y) {
  return f.fun(y)+y;
}
