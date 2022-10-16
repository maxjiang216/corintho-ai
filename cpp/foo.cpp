#include "foobar.h"

Foo::Foo(int x): x{x} {}

int Foo::fun(int y) {
  return x+y;
}
