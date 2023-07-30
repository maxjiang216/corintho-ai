# distutils: language = c++

import time

import numpy as np
import tflite_runtime.interpreter as tflite

from libcpp cimport bool

_NUM_MOVES = 96
_GAME_STATE_SIZE = 70

cdef extern from "../cpp/src/tourney.cpp":
    cdef cppclass Tourney:
        Tourney(int num_threads) except +
        bool all_done() except +
        int num_requests(int id) except +
        void writeScores(string filename) except +
        void writeRequests(float *game_states, int id) except +
        void doIteration(float *eval, float *probs) except +
        void addPlayer(
            int player_id,
            int model_id,
            int max_searches,
            int searches_per_eval,
            float c_puct,
            float epsilon,
            bool random) except +
        void addMatch(int player1, int player2) except +

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

cpdef get_tourney(Tourney *tourney, filename):
    """
    Returns a pointer to a Tourney object
    Reads player and match pairings from filename
    """

    searches_per_eval = {}
    player_models = {}
    max_model_searches = {}
    model_ids = set()

    with open(filename, "r") as f:
        # Get players
        num_players = int(f.readline())
        for i in range(num_players):
            player_id, model_id, max_searches, searches_per_eval, c_puct, epsilon, random = [
                float(x) for x in f.readline().split()
            ]
            tourney.addPlayer(
                int(player_id),
                int(model_id),
                int(max_searches),
                int(searches_per_eval),
                float(c_puct),
                float(epsilon),
                bool(random),
            )
            searches_per_eval[int(player_id)] = int(searches_per_eval)
            player_models[int(player_id)] = int(model_id)
            model_ids.add(int(model_id))

        # Get matches
        num_matches = int(f.readline())
        for i in range(num_matches):
            player1, player2 = [int(x) for x in f.readline().split()]
            tourney.addMatch(player1, player2)

            # Too much effort for the edge case that they are equal
            if player_models[player1] not in max_model_searches:
                max_model_searches[player_models[player1]] = searches_per_eval[player1]
            else:
                max_model_searches[player_models[player1]] += searches_per_eval[player1]
            if player_models[player2] not in max_model_searches:
                max_model_searches[player_models[player2]] = searches_per_eval[player2]
            else:
                max_model_searches[player_models[player2]] += searches_per_eval[player2]

        return model_ids, max(max_model_searches.values())

cdef do_game_loop(Tourney *tourney, models, model_ids, max_searches, log_folder):
    """
    Main game loop
    """

    cdef np.ndarray[np.float32_t, ndim=1] evals = np.zeros(max_searches, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] probs = np.zeros((max_searches, _NUM_MOVES), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] game_states = np.zeros((max_searches, _GAME_STATE_SIZE), dtype=np.float32)
    predict_time = 0.0
    play_time = 0.0
    evals_done = 0
    start_time = time.perf_counter()
    last_time = start_time

    while not tourney.all_done():

        for id in model_ids:
            num_requests = tourney.num_requests(id)

            if num_requests == 0:
                continue
            
            tourney.writeRequests(&game_states[0,0], id)

            pred_start = time.perf_counter()
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
            probs = model.get_tensor(output_details[1]['index'])
            predict_time += time.perf_counter() - pred_start

            play_start = time.perf_counter()
            tourney.doIteration(eval, probs, id)
            play_time += time.perf_counter() - play_start

        evals_done += 1

        if time.perf_counter() - last_time > 60:
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

            
def play_games(model_paths, filename, num_threads, log_folder):
    """
    models is a list of model paths (TFLite)
    Reads pairings from filename
    """

    # Load models
    models = get_models(model_paths)

    # Create tournament and load pairings
    Tourney *tourney = new Tourney(num_threads)
    model_ids, max_searches = get_tourney(tourney, filename)

    # Play games
    do_game_loop(tourney, models, model_ids, max_searches, log_folder)

    # Save results
    tourney.writeScores(f"{log_folder}/scores.txt")