# distutils: language = c++

cdef extern from "move.cpp":
    cdef cppclass Move:
        bint mtype
        short ptype, row1, col1, row2, col2
        Move() except + 
        Move(short) except +
    Move get_move_from_id(short) except +
    short encode_place(short, short, short) except +
    short encode_move(short, short, short, short) except +

# Test functionality
cdef Move x = get_move_from_id(0)
print(x.ptype)
