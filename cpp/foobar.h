#ifndef FOOBAR_H
#define FOOBAR_H

class Foo {
    int x;
  public:
    Foo(int);
    int fun(int);
};

class Bar {
    Foo f;
  public:
    Bar(int);
    int fun(int);
};

#endif
