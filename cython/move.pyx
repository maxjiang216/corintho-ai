cpdef Move move_from_id(int move_id):
    cdef Move move
    # Move
    move.mtype = False
    # Right
    if move_id < 12:
        move.row1 = move_id // 3
        move.col1 = move_id % 3
        move.row2 = move.row1
        move.col2 = move.col1 + 1
    # Down
    elif move_id < 24:
        move.row1 = (move_id - 12) // 4
        move.col1 = move_id % 4
        move.row2 = move.row1 + 1
        move.col2 = move.col1
    # Left
    elif move_id < 36:
        move.row1 = (move_id - 24) // 3
        move.col1 = move_id % 3 + 1
        move.row2 = move.row1
        move.col2 = move.col1 - 1
    # Up
    elif move_id < 48:
        move.row1 = (move_id - 36) // 4 + 1
        move.col1 = move_id % 4
        move.row2 = move.row1 - 1
        move.col2 = move.col1
    # Place
    else:
        move.mtype = True
        move.ptype = (move_id - 48) // 16
        move.row1 = (move_id % 16) // 4
        move.col1 = move_id % 4
    
    return move

cpdef Move get_place(int ptype, int row, int col):
    cdef Move move
    move.mtype = True
    move.ptype = ptype
    move.row1 = row
    move.col1 = col
    return move

cpdef Move get_move(int row1, int col1, int row2, int col2):
    cdef Move move
    move.mtype = False
    move.row1 = row1
    move.col1 = col1
    move.row2 = row2
    move.col2 = col2
    return move

cpdef int encode_place(int ptype, int row, int col):
    return 48 + ptype * 16 + row * 4 + col

cpdef int encode_move(int row1, int col1, int row2, int col2):
    # Right
    if col1 < col2:
        return row1 * 3 + col1
    # Down
    if row1 < row2:
        return 12 + row1 * 4 + col1
    # Left
    if col1 > col2:
        return 24 + row1 * 3 + (col1 - 1)
    # Up
    return 36 + (row1 - 1) * 4 + col1

cpdef str move_to_str(Move move):
    output = ""
    # Place
    if move.mtype:
        output += "P"
        if move.ptype == 0:
            output += "B"
        elif move.ptype == 1:
            output += "C"
        elif move.ptype == 2:
            output += "A"
        output += str(move.row1)
        output += str(move.col1)
        return output
    # Move
    output += "M"
    output += str(move.row1)
    output += str(move.col1)
    if move.row2 - move.row1 == 1:
        output += "D"
    elif move.row2 - move.row1 == -1:
        output += "U"
    elif move.col2 - move.col1 == 1:
        output += "R"
    elif move.col2 - move.col1 == -1:
        output += "L"
    return output
