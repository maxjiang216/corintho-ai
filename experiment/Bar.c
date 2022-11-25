#include "Foo.c"

struct Bar {
	struct Foo *foo;
};

struct Bar* get_bar(int x) {
	struct Foo *foo = malloc(sizeof(struct Foo));
	foo->x = x;
	struct Bar *bar = malloc(sizeof(struct Bar));
	bar->foo = foo;
	return bar;
}

int do_bar(struct Bar *bar, int x) {
	return bar->foo->x + x;
}
