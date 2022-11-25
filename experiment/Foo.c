struct Foo {
	int x;
};

int do_foo(struct Foo *foo, int x) {
	return foo->x + x;
}
