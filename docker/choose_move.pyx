# distutils: language = c++

import time
from libcpp cimport bool
import numpy as np
cimport numpy as np

cdef extern from "../cpp/move.cpp":
    cdef cppclass Move:
        Move(int move)
        bool mtype
        int ptype
        int row1
        int col1
        int row2
        int col2

cdef extern from "../cpp/playmc.cpp":
    cdef cppclass PlayMC:
        PlayMC(
            bool *board,
            int to_play,
            int *pieces,
            int searches_per_eval,
            int seed
        )
        bool do_iteration(float *evaluations, float *probabilities)
        int write_requests(float *game_states)
        int choose_move()
        void get_legal_moves(int *legal_moves)
        int get_node_number()
        bool is_done()
        bool has_drawn()

def choose_move(
        game_state,
        time_limit,
        searches_per_eval=1,
        max_nodes=0,
    ):
    """
    Use MCST and neural network to choose a move.
    game_state is dictionary of game state
    time_limit is time limit in seconds
    max_searches is maximum number of searches per evaluation (0 for no limit)
    num_threads is number of threads to use (0 for no limit)
    max_nodes is maximum number of nodes to search (0 for no limit)
    """

    # Constants
    cdef int NUM_MOVES = 96
    cdef int GAME_STATE_SIZE = 70
    MODEL_LOCATION = "."
    
    start_time = time.time()

    # Extract game state
    bool *board = new bool[64]
    for i, row in enumerate(game_state["board"]):
        for j, space in enumerate(row):
            board[(i * 8 + j) * 4] = space["pieces"]["base"]
            board[(i * 8 + j) * 4 + 1] = space["pieces"]["column"]
            board[(i * 8 + j) * 4 + 2] = space["pieces"]["capital"]
            board[(i * 8 + j) * 4 + 3] = space["frozen"]
    int to_play = game_state["turn"]
    int *pieces = new int[6]
    for i, pieceType in enumerate(["base", "column", "capital"]):
        pieces[i] = game_state["players"]["pieceCounts"][pieceType]

    rng = np.random.default_rng(int(start_time))

    # Construct a MCST object
    cdef PlayMC *mcst = new PlayMC(
        board,
        to_play,
        *pieces,
        searches_per_eval,
        rng.integers(65536),
    )

    if mcst.is_done():
        if mcst.has_drawn():
            return {"pre-result": "draw"}
        else:
            # Human player has won
            return {"pre-result": "win"}

    # Load playing model
    model = load_model(MODEL_LOCATION)

    cdef np.ndarray[np.float32_t, ndim=1] evaluations = np.zeros(searches_per_eval, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] probabilities = np.zeros((searches_per_eval, NUM_MOVES), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] game_states = np.zeros((searches_per_eval, GAME_STATE_SIZE), dtype=np.float32)
    
    # While the game is not done
    while not mcst.is_done():
        # Do some iterations of the MCST
        res = mcst.do_iteration(&evaluations[0], &probabilities[0,0])
        
        # This means that the MCST has deduced the game outcome
        if res:
            break

        # Stop conditions
        if time.time() - start_time > time_limit or (max_nodes > 0 and mcst.get_node_number() > max_nodes):
            break

        num_requests = mcst.write_requests(&game_states[0,0])
        if num_requests == 0:
            break

        # Evaluate with neural network
        res = model.predict(
            x=game_states, verbose=0,
        )
        evaluations = res[0].flatten()
        probabilities = res[1]

    # Choose the best move
    Move move = Move(mcst.choose_move())
    if move.mtype:
        move = {
            "mtype": "place",
            "piecetype": move.ptype,
            "row": move.row1,
            "col": move.col1,
        }
    else:
        move = {
            "mtype": "move",
            "sourceRow": move.row1,
            "sourceCol": move.col1,
            "targetRow": move.row2,
            "targetCol": move.col2,
        }
    is_done = mcst.is_done()
    has_won = False
    if is_done:
        has_won = not mcst.has_drawn()
    legal_move_list = []
    if not is_done:
        bool *legal_moves = new bool[NUM_MOVES]
        mcst.get_legal_moves(legal_moves)
        for i in range(NUM_MOVES):
            if legal_moves[i]:
                legal_move_list.append(i)

    return {
        "move": move,
        "is_done": is_done,
        "has_won": has_won,
        "legal_moves": legal_move_list,
    }
