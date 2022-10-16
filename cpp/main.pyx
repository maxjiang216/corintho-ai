cdef extern from "foobar.h":
    cdef cppclass Foo:
        Foo(int) except +
        fun(int) except +
    cdef cppclass Bar:
        Bar(int) except +
        fun(int) except +
