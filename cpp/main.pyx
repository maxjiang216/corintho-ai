# distutils: language = c++

cdef extern from "move.cpp":
    cdef cppclass Move:
        bint mtype
        int ptype, row1, col1, row2, col2
        Move()
    Move get_move_from_id(int) except +
    short encode_place(int, int, int) except +
    short encode_move(int, int, int, int) except +

# Test functionality
cdef Move x = get_move_from_id(48)
print(x.ptype)
