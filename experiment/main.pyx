cdef extern from "Bar.c":
    cdef struct Bar:
        pass
    Bar *get_bar(int x)
    int do_bar(Bar *bar, int x)

cdef Bar *bar = get_bar(5)
print(do_bar(bar, 1))
