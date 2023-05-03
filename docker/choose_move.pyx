# distutils: language = c++

import time
from libcpp cimport bool
import numpy as np
cimport numpy as np

from keras.api._v2.keras.models import load_model

cdef extern from "../cpp/playmc.cpp":
    cdef cppclass PlayMC:
        PlayMC(
            long *board,
            int to_play,
            long *pieces,
            int searches_per_eval,
            int seed
        )
        bool do_iteration(float *evaluations, float *probabilities)
        int write_requests(float *game_states)
        int choose_move()
        void get_legal_moves(long *legal_moves)
        int get_node_number()
        bool is_done()
        bool has_drawn()

# Constants
cdef int NUM_MOVES = 96
cdef int GAME_STATE_SIZE = 70
pieceTypes = ["base", "column", "capital"]
MODEL_LOCATION = "model"

# Load playing model
model = load_model(MODEL_LOCATION)

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

    
    
    start_time = time.time()

    # Extract game state
    cdef long[:] board = np.zeros(64, dtype=long)
    for i, row in enumerate(game_state["board"]):
        for j, space in enumerate(row):
            board[(i * 4 + j) * 4] = 1 if space["pieces"]["base"] else 0
            board[(i * 4 + j) * 4 + 1] = 1 if space["pieces"]["column"] else 0
            board[(i * 4 + j) * 4 + 2] = 1 if space["pieces"]["capital"] else 0
            board[(i * 4 + j) * 4 + 3] = 1 if space["frozen"] else 0
    cdef int to_play = game_state["turn"]
    cdef long[:] pieces = np.zeros(6, dtype=long)
    for i, pieceType in enumerate(pieceTypes):
        pieces[i] = game_state["players"][0]["pieceCounts"][pieceType]
        pieces[3 + i] = game_state["players"][1]["pieceCounts"][pieceType]

    rng = np.random.default_rng(int(start_time))

    # Construct a MCST object
    cdef PlayMC *mcst = new PlayMC(
        &board[0],
        to_play,
        &pieces[0],
        searches_per_eval,
        rng.integers(65536),
    )

    if mcst.is_done():
        if mcst.has_drawn():
            return {"pre-result": "draw"}
        else:
            # Human player has won
            return {"pre-result": "win"}

    cdef np.ndarray[np.float32_t, ndim=1] evaluations = np.zeros(searches_per_eval, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] probabilities = np.zeros((searches_per_eval, NUM_MOVES), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] game_states = np.zeros((searches_per_eval, GAME_STATE_SIZE), dtype=np.float32)

    counter = 0
    # While the game is not done
    while counter < 2 or (time.time() - start_time < time_limit and (max_nodes == 0 or mcst.get_node_number() <= max_nodes)):
        counter += 1
        # Do some iterations of the MCST
        res = mcst.do_iteration(&evaluations[0], &probabilities[0,0])
        
        # This means that the MCST has deduced the game outcome
        if res:
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
    move = mcst.choose_move()
    is_done = mcst.is_done()
    has_won = False
    if is_done:
        has_won = not mcst.has_drawn()
    legal_move_list = []
    cdef long[:] legal_moves = np.zeros(NUM_MOVES, dtype=long)
    if not is_done:
        mcst.get_legal_moves(&legal_moves[0])
        for i in range(NUM_MOVES):
            if legal_moves[i] == 1:
                legal_move_list.append(i)

    return {
        "move": move,
        "is_done": is_done,
        "has_won": has_won,
        "legal_moves": legal_move_list,
    }
