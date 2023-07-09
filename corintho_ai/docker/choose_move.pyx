"""
Module for choosing moves using MCST and a neural network
for the Corintho web app.

choose_move: Main Cython function called by the Flask API.
"""

# distutils: language = c++

import sys
import time

import numpy as np
from keras.api._v2.keras.models import load_model

cimport numpy as np
from libcpp cimport bool


cdef extern from "<random>" namespace "std":
    cdef cppclass mt19937:
        mt19937() except +
        mt19937(unsigned int seed) except +

cdef extern from "../cpp/src/trainmc.cpp":
    cdef cppclass TrainMC:
        TrainMC(
            mt19937 *generator,
            float *to_eval,
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
    for i, row in enumerate(game_state["board"]):
        for j, space in enumerate(row):
            board[(i * 4 + j) * 4] = 1 if space["pieces"]["base"] else 0
            board[(i * 4 + j) * 4 + 1] = 1 if space["pieces"]["column"] else 0
            board[(i * 4 + j) * 4 + 2] = 1 if space["pieces"]["capital"] else 0
            board[(i * 4 + j) * 4 + 3] = 1 if space["frozen"] else 0
    cdef int to_play = game_state["turn"]
    for i, piece_type in enumerate(_PIECE_TYPES):
        pieces[i] = game_state["players"][0]["pieceCounts"][piece_type]
        pieces[3 + i] = game_state["players"][1]["pieceCounts"][piece_type]

    return to_play

cdef dict get_pre_result(TrainMC* mc):
    """
    Get "pre-result" from MCST.

    A "pre-result" means the game is over after the human move (and before the AI move).

    mc: pointer to TrainMC object

    Returns: dictionary of pre-result (or None if no pre-result)
    """
    if mc.done():
        if mc.drawn():
            return {"pre-result": "draw"}
        # Human player has won
        return {"pre-result": "win"}
    return None

cdef void search(TrainMC* mc, model, searches_per_eval, time_limit, start_time):
    """
    Do a search with the MCST.

    Continues searching until the time limit is reached or the maximum number of searches is reached.

    mc: pointer to TrainMC object
    model: TFLite model
    """
    # Get neural network input and output shapes
    input_details = model.get_input_details()
    output_details = model.get_output_details()
    input_shape = input_details[0]['shape']

    print("Before allocating numpy arrays")
    # Arrays for input and output with the neural network
    cdef np.ndarray[np.float32_t, ndim=2] game_states = np.zeros((searches_per_eval, _GAME_STATE_SIZE), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] eval = np.zeros(searches_per_eval, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] probs = np.zeros((searches_per_eval, _NUM_MOVES), dtype=np.float32)
    print("After allocating numpy arrays")

    evals_done = 0
    # We continue searching until the time limit is reached or the maximum number of searches is reached.
    # We do at least 2 searches
    while evals_done < 3 or time.time() - start_time < time_limit:
        evals_done += 1
        done = mc.doIteration(&eval[0], &probs[0,0])
        print("After doIteration")
        # The MCST has deduced the game outcome or a maximum number of searches is reached
        if done:
            break
        print("Before writing requests")
        # Get requests for evaluation
        num_requests = mc.num_requests()
        print("Number of requests: ", num_requests)
        if num_requests == 0:
            break
        mc.writeRequests(&game_states[0,0])
        print("After writing requests and before converting to float32")
        input_data = game_states[:num_requests].astype(np.float32)
        # Resize the input tensor depending on the number of requests
        # This is needed in TFLite models
        input_shape = list(input_details[0]['shape'])
        input_shape[0] = num_requests
        print("Before resizing input tensor")
        model.resize_tensor_input(input_details[0]['index'], input_shape)
        print("After resizing input tensor and before allocating tensors")
        model.allocate_tensors()
        print("After allocating tensors and before setting tensor")
        model.set_tensor(input_details[0]['index'], input_data)
        print("After setting tensor and before invoking model")
        model.invoke()
        print("After invoking model and before getting tensor")
        eval = model.get_tensor(output_details[0]['index']).flatten()
        prob = model.get_tensor(output_details[1]['index'])
        print("After getting tensor")

cdef list get_legal_moves(TrainMC* mc):
    """
    Get list of legal moves from MCST.

    This allows the Javascript in the web app easily check if a move is legal.

    mc: pointer to TrainMC object

    Returns: list of IDs of legal moves
    """
    legal_move_lst = []
    cdef int[:] legal_moves = np.zeros(_NUM_MOVES, dtype=np.int32)
    mc.getLegalMoves(&legal_moves[0])
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
    cdef mt19937 *generator = new mt19937(rng.integers(65536))
    cdef float[:] to_eval = np.zeros(searches_per_eval, dtype=np.float32)
    if max_searches == 0:
        max_searches = 32760  # So that visit count fits in a 16-bit signed integer
    cdef TrainMC *mc = new TrainMC(
        generator,
        &to_eval[0],
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
        legal_moves = get_legal_moves(mc)
    nodes_searched = mc.num_nodes()
    evaluation = mc.eval()
    del generator
    del mc

    return {
        "move": move,
        "is_done": is_done,
        "has_won": has_won,
        "legal_moves": legal_moves,
        "nodes_searched": nodes_searched,
        "evaluation": evaluation / nodes_searched,
    }
