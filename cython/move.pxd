cdef struct Move:
    bint mtype
    int row1, col1, row2, col2, ptype

cpdef Move move_from_id(int move_id)
cpdef Move get_place(int ptype, int row, int col)
cpdef Move get_move(int row1, int col1, int row2, int col2)
cpdef int encode_place(int ptype, int row, int col)
cpdef int encode_move(int row1, int col1, int row2, int col2)
cpdef str move_to_str(Move move)

