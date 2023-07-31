# distutils: language = c++

import time

import numpy as np
import tflite_runtime.interpreter as tflite

cimport numpy as np
from libcpp cimport bool
from libcpp.string cimport string

_NUM_MOVES = 96
_GAME_STATE_SIZE = 70

cdef extern from "../cpp/src/tourney.cpp":
    cdef cppclass Tourney:
        Tourney(int num_threads, string log_folder) except +
        bool all_done() except +
        int num_requests(int id) except +
        void writeScores(string filename) except +
        void writeRequests(float *game_states, int id) except +
        void doIteration(float *eval, float *probs, int id) except +
        void addPlayer(
            int player_id,
            int model_id,
            int max_searches,
            int searches_per_eval,
            float c_puct,
            float epsilon,
            bool random) except +
        void addMatch(int player1, int player2, bool logging) except +

cpdef format_time(t):
    """Format string
    t is time in seconds"""
    if t < 1:
        return f"{t}s"
    if t < 60:
        return f"{t:.2f}s"
    if t < 3600:
        return f"{int(t//60)}m{round(t)%60:02d}s"
    return f"{int(t//3600)}h{int((t % 3600)//60):02d}m{round(t)%60:02d}s"

cpdef get_models(model_paths):
    """
    model_paths is a list of paths to TFLite models
    Returns a list of TFLite models
    """

    # Load models
    models = [
        tflite.Interpreter(model_path=model_path)
        for model_path in model_paths
    ]

    # Allocate tensors
    for model in models:
        model.allocate_tensors()

    return models

cdef get_tourney(Tourney *tourney, player_file, match_file):
    """
    Returns a pointer to a Tourney object
    Reads player and match pairings from filename
    """

    searches_per_evals = {}
    player_models = {}
    max_model_searches = {}
    model_ids = set()

    with open(player_file, "r") as f:
        # Get players
        num_players = int(f.readline())
        for i in range(num_players):
            player_id, model_id, max_searches, searches_per_eval, c_puct, epsilon, random = [
                int(x) if i != 4 and i != 5 else float(x) for i, x in enumerate(f.readline().split())
            ]
            tourney.addPlayer(
                player_id,
                model_id,
                max_searches,
                searches_per_eval,
                c_puct,
                epsilon,
                True if random == 1.0 else False,
            )
            searches_per_evals[int(player_id)] = int(searches_per_eval)
            player_models[int(player_id)] = int(model_id)
            model_ids.add(int(model_id))

    # Get 
    with open(match_file, "r") as f:
        num_matches = int(f.readline())
        for i in range(num_matches):
            player1, player2, logging = [int(x) for x in f.readline().split()]
            tourney.addMatch(player1, player2, True if logging == 1 else False)

            # Too much effort for the edge case that they are equal
            if player_models[player1] not in max_model_searches:
                max_model_searches[player_models[player1]] = searches_per_evals[player1]
            else:
                max_model_searches[player_models[player1]] += searches_per_evals[player1]
            if player_models[player2] not in max_model_searches:
                max_model_searches[player_models[player2]] = searches_per_evals[player2]
            else:
                max_model_searches[player_models[player2]] += searches_per_evals[player2]

        return model_ids, max(max_model_searches.values())

cdef play_games(Tourney *tourney, models, model_ids, max_searches, log_folder):
    """
    Main game loop
    """

    cdef np.ndarray[np.float32_t, ndim=1] eval = np.zeros(max_searches, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] probs = np.zeros((max_searches, _NUM_MOVES), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] game_states = np.zeros((max_searches, _GAME_STATE_SIZE), dtype=np.float32)
    predict_time = 0.0
    play_time = 0.0
    evals_done = 0
    start_time = time.perf_counter()
    last_time = start_time

    while not tourney.all_done():

        for id in model_ids:

            # Dummy model IDs for random players are negative
            num_requests = 0
            if id >= 0:
                num_requests = tourney.num_requests(id)

            if num_requests > 0:
            
                tourney.writeRequests(&game_states[0,0], id)

                pred_start = time.perf_counter()
                input_data = game_states[:num_requests].astype(np.float32)
                # Resize the input tensor depending on the number of requests
                # This is needed in TFLite models
                model = models[id]
                input_details = model.get_input_details()
                output_details = model.get_output_details()
                input_shape = input_details[0]['shape']
                input_shape = list(input_details[0]['shape'])
                input_shape[0] = num_requests
                model.resize_tensor_input(input_details[0]['index'], input_shape)
                model.allocate_tensors()
                model.set_tensor(input_details[0]['index'], input_data)
                model.invoke()
                eval = model.get_tensor(output_details[0]['index']).flatten()
                probs = model.get_tensor(output_details[1]['index'])
                predict_time += time.perf_counter() - pred_start

            play_start = time.perf_counter()
            tourney.doIteration(&eval[0], &probs[0,0], id)
            play_time += time.perf_counter() - play_start

        evals_done += 1

        if time.perf_counter() - last_time > 60 or tourney.all_done():
            time_taken = time.perf_counter() - start_time
            with open(f"{log_folder}/progress.txt", 'a+', encoding='utf-8') as f:
                f.write(
                    f"{evals_done} evaluations completed in {format_time(time_taken)}\n"
                    f"Predicted time to complete: "
                    f"{format_time(100 * 26.5 * time_taken / evals_done)}\n"
                    f"Neural network prediction time so far: {format_time(predict_time)}\n"
                    f"Self play time so far: {format_time(play_time)}\n\n"
                )
            last_time = time.perf_counter()

            
def run(model_paths, filename, num_threads, log_folder):
    """
    models is a list of model paths (TFLite)
    Reads pairings from filename
    """

    # Load models
    models = get_models(model_paths)

    # Create tournament and load pairings
    cdef Tourney *tourney = new Tourney(num_threads, log_folder.encode())
    model_ids, max_searches = get_tourney(tourney, filename)

    # Play games
    play_games(tourney, models, model_ids, max_searches, log_folder)

    # Save results
    tourney.writeScores(f"{log_folder}/scores.txt".encode())

    del tourney