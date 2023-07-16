"""
Module for choosing moves using MCST and a neural network
for the Corintho web app.

choose_move: Main Cython function called by the Flask API.
"""

# distutils: language = c++

import time

import numpy as np

cimport numpy as np
from libcpp cimport bool


cdef extern from "../cpp/src/dockermc.cpp":
    cdef cppclass DockerMC:
        DockerMC(
            int seed,
            int max_searches,
            int searches_per_eval,
            float c_puct,
            float epsilon,
            int *board,
            int to_play,
            int *pieces,
        ) except +
        float eval() except +
        int num_requests() except +
        int num_nodes() except +
        bool done() except +
        bool drawn() except +
        void writeRequests(float *game_states) except +
        void getLegalMoves(int *legal_moves) except +
        int chooseMove() except +
        bool doIteration(float *eval, float *probs) except +

# Constants
cdef int _NUM_MOVES = 96
cdef int _GAME_STATE_SIZE = 70
_PIECE_TYPES = ["base", "column", "capital"]

cdef int extract_game_state(game_state, int[:] board, int[:] pieces):
    """
    Extract game state from game_state dictionary.

    game_state: dictionary of game state
    
    Returns: tuple of (board, to_play, pieces)
    """
    cdef int i, j
    for i in range(4):  # Assuming game_state["board"] is a 4x4 list
        for j in range(4):  # Assuming each row is a list of 4 elements
            space = game_state["board"][i][j]
            board[(i * 4 + j) * 4] = 1 if space["pieces"]["base"] else 0
            board[(i * 4 + j) * 4 + 1] = 1 if space["pieces"]["column"] else 0
            board[(i * 4 + j) * 4 + 2] = 1 if space["pieces"]["capital"] else 0
            board[(i * 4 + j) * 4 + 3] = 1 if space["frozen"] else 0

    cdef int to_play = game_state["turn"]
    for i in range(3):  # Assuming there are 6 piece types
        piece_type = _PIECE_TYPES[i]
        pieces[i] = game_state["players"][0]["pieceCounts"][piece_type]
        pieces[3 + i] = game_state["players"][1]["pieceCounts"][piece_type]

    return to_play

cdef dict get_pre_result(DockerMC* mc):
    """
    Get "pre-result" from MCST.

    A "pre-result" means the game is over after the human move (and before the AI move).

    mc: pointer to DockerMC object

    Returns: dictionary of pre-result (or None if no pre-result)
    """
    if mc.done():
        if mc.drawn():
            return {"pre-result": "draw"}
        # Human player has won
        return {"pre-result": "win"}
    return None

cdef void search(DockerMC* mc, model, searches_per_eval, time_limit, start_time):
    """
    Do a search with the MCST.

    Continues searching until the time limit is reached or the maximum number of searches is reached.

    mc: pointer to DockerMC object
    model: TFLite model
    """
    # Get neural network input and output shapes
    input_details = model.get_input_details()
    output_details = model.get_output_details()
    input_shape = input_details[0]['shape']

    # Arrays for input and output with the neural network
    cdef np.ndarray[np.float32_t, ndim=2] game_states = np.zeros((searches_per_eval, _GAME_STATE_SIZE), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] eval = np.zeros(searches_per_eval, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] probs = np.zeros((searches_per_eval, _NUM_MOVES), dtype=np.float32)

    evals_done = 0
    # We continue searching until the time limit is reached or the maximum number of searches is reached.
    # We do at least 2 searches
    while evals_done < 3 or time.time() - start_time < time_limit:
        evals_done += 1
        done = mc.doIteration(&eval[0], &probs[0,0])
        # The MCST has deduced the game outcome or a maximum number of searches is reached
        if done:
            break
        # Get requests for evaluation
        num_requests = mc.num_requests()
        if num_requests == 0:
            break
        mc.writeRequests(&game_states[0,0])
        input_data = game_states[:num_requests].astype(np.float32)
        # Resize the input tensor depending on the number of requests
        # This is needed in TFLite models
        input_shape = list(input_details[0]['shape'])
        input_shape[0] = num_requests
        model.resize_tensor_input(input_details[0]['index'], input_shape)
        model.allocate_tensors()
        model.set_tensor(input_details[0]['index'], input_data)
        model.invoke()
        eval = model.get_tensor(output_details[0]['index']).flatten()
        prob = model.get_tensor(output_details[1]['index'])

cdef list get_legal_moves(DockerMC* mc):
    """
    Get list of legal moves from MCST.

    This allows the Javascript in the web app easily check if a move is legal.

    mc: pointer to DockerMC object

    Returns: list of IDs of legal moves
    """
    legal_move_lst = []
    cdef int[:] legal_moves = np.zeros(_NUM_MOVES, dtype=np.int32)
    mc.getLegalMoves(&legal_moves[0])
    cdef int i
    for i in range(_NUM_MOVES):
        if legal_moves[i] == 1:
            legal_move_lst.append(i)
    return legal_move_lst
   

def choose_move(
        game_state,
        time_limit,
        model,
        searches_per_eval=1,
        max_searches=0,
    ):
    """
    Use the MCST and neural network algorithm to choose a move.

    game_state: dictionary of the game state.
    time_limit: the time limit for the search in seconds
    searches_per_eval: number of searches per neural network evaluation
    max_searches: the maximum number of searches to do (0 for no limit)
    model: TFLite model
    """
    
    start_time = time.time()
    rng = np.random.default_rng(int(start_time))

    # Extract game state
    cdef int[:] board = np.zeros(64, dtype=np.int32)
    cdef int[:] pieces = np.zeros(6, dtype=np.int32)
    to_play = extract_game_state(game_state, board, pieces)

    # Construct a MCST object
    if max_searches == 0:
        max_searches = 32760  # So that visit count fits in a 16-bit signed integer
    cdef DockerMC *mc = new DockerMC(
        rng.integers(65536),
        max_searches,
        searches_per_eval,
        1.0,
        0.25,
        &board[0],
        to_play,
        &pieces[0],
    )

    pre_result = get_pre_result(mc)
    if pre_result:
        del mc
        return pre_result

    # Search with the MCST
    search(mc, model, searches_per_eval, time_limit, start_time);
    
    # Choose the best move and get other information
    move = mc.chooseMove()
    is_done = mc.done()
    has_won = False
    legal_moves = []
    if is_done:
        has_won = not mc.drawn()
    else:
        legal_moves = get_legal_moves(mc)
    nodes_searched = mc.num_nodes()
    evaluation = mc.eval()
    del mc

    return {
        "move": move,
        "is_done": is_done,
        "has_won": has_won,
        "legal_moves": legal_moves,
        "nodes_searched": nodes_searched,
        "evaluation": evaluation / nodes_searched,
    }
