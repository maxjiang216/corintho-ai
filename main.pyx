# distutils: language = c++

from libcpp.string cimport string
from libcpp cimport bool
import numpy as np
cimport numpy as np
import time
import datetime
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import keras.api._v2.keras as keras
from keras.api._v2.keras.models import load_model
from keras import backend as K

cdef extern from "cpp/trainer.cpp":
    cdef cppclass Trainer:
        Trainer()
        Trainer(int num_games, int num_logged, int num_iterations,
                float c_puct, float epsilon, string logging_folder, int random_seed)
        Trainer(int num_games, int num_logged, int num_iterations,
                float c_puct, float epsilon, string logging_folder, int random_seed, bool)
        bool do_iteration(float *evaluations, float *probabilities, float *dirichlet,
                          float *game_states)
        bool do_iteration(float *evaluations_1, float *probabilities_1,
                          float *evaluations_2, float *probabilities_2,
                          float *dirichlet_noise, float *game_states)
        int count_samples()
        void write_samples(float *game_states, float *evaluation_samples, float *probability_samples)
        float get_score()

cdef extern from "cpp/manager.cpp":
    cdef cppclass Manager:
        Manager()
        Manager(int num_games, int num_logged, int num_iterations,
                float c_puct, float epsilon, string logging_folder, int random_seed, int processes)
        bool do_iteration(float *evaluations, float *probabilities, float *dirichlet,
                          float *game_states)
        int count_samples()
        void write_samples(float *game_states, float *evaluation_samples, float *probability_samples)

def format_time(t):
    """Format string
    t is time in seconds"""

    if t < 60:
        return f"{t:.1f} seconds"
    if t < 3600:
        return f"{t/60:.1f} minutes"
    return f"{t/60/60:.1f} hours"

def train_generation(*,
    cur_gen_location,  # neural network to train on
    best_gen_location,  # opponent to test against
    new_model_location,  # location to save new model (is this a folder or a file?)
    train_log_folder,
    test_log_folder,
    train_sample_folder,  # String path to write training samples into
    unsigned int num_games=25000,
    iterations=1600,
    num_test_games=400,
    num_logged=10,
    testing_threshold=0.5,  # Non-inclusive lower bound for new generation to pass
    c_puct=1.0,
    epsilon=0.25,
    processes=1,
    learning_rate=0.01,
    batch_size=2048,
    epochs=1,
    old_training_samples=[],  # list of folders containing training sample files from previous generations to use
):

    cdef unsigned int NUM_TOTAL_MOVES = 96
    cdef unsigned int NUM_MOVES = 96
    cdef unsigned int GAME_STATE_SIZE = 70

    rng = np.random.default_rng(int(time.time()))

    # Training

    # Load playing model
    model = load_model(best_gen_location)

    '''
    cdef Trainer* trainer = new Trainer(
        num_games,
        num_logged,
        iterations,
        c_puct,
        epsilon,
        train_log_folder.encode(),
        rng.integers(65536),  # Random seed
    )'''
    cdef Manager* trainer = new Manager(
        num_games,
        num_logged,
        iterations,
        c_puct,
        epsilon,
        train_log_folder.encode(),
        rng.integers(65536),  # Random seed
        processes,
    )

    cdef np.ndarray[np.float32_t, ndim=1] evaluations = np.zeros(num_games, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] probabilities = np.zeros((num_games, NUM_TOTAL_MOVES), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] dirichlet = np.zeros(num_games*NUM_MOVES, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] game_states = np.zeros((num_games, GAME_STATE_SIZE), dtype=np.float32)

    cdef int evaluations_done = 0

    print(f"Begin training!")

    start_time = time.perf_counter()

    # First iteration
    trainer.do_iteration(&evaluations[0], &probabilities[0,0], &dirichlet[0], &game_states[0,0])

    predict_time = 0
    play_time = 0
    d_time = 0

    while True:

        pred_start = time.perf_counter()
        res = model.predict(
            x=game_states, batch_size=num_games, verbose=0, use_multiprocessing=True
        )
        predict_time += time.perf_counter()-pred_start
        evaluations = res[0].flatten()
        probabiltiies = res[1]

        d_start = time.perf_counter()
        dirichlet = rng.dirichlet((0.3,), num_games*NUM_MOVES).reshape((num_games*NUM_MOVES,)).astype(np.float32)
        d_time += time.perf_counter() - d_start
        play_start = time.perf_counter()
        res = trainer.do_iteration(&evaluations[0], &probabilities[0,0], &dirichlet[0], &game_states[0,0])
        play_time += time.perf_counter()-play_start
        if res:
            print("Done all")
            break
        evaluations_done += 1
        if evaluations_done % max(1, 15 * iterations // 1000) == 0:
            time_taken = time.perf_counter() - start_time
            print(
                f"{evaluations_done} evaluations completed in {format_time(time_taken)}\n"
                f"Predicted time to complete: {format_time(26.67*iterations*time_taken/evaluations_done)}\n"
                f"Estimated time left: {format_time((26.67*iterations-evaluations_done)*time_taken/evaluations_done)}\n"
                f"Prediction time so far: {format_time(predict_time)}\n"
                f"Play time so far: {format_time(play_time)}\n"
                f"Dirichlet time so far: {format_time(d_time)}\n"
            )

    # Get training samples
    num_samples = trainer.count_samples()
    cdef np.ndarray[np.float32_t, ndim=2] sample_states = np.zeros((num_samples, GAME_STATE_SIZE), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] evaluation_labels = np.zeros(num_samples, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] probability_labels = np.zeros((num_samples, NUM_TOTAL_MOVES), dtype=np.float32)
    trainer.write_samples(&sample_states[0,0], &evaluation_labels[0], &probability_labels[0,0])

    del trainer

    # Save training samples
    np.save(f"{train_sample_folder}/game_states", sample_states)
    np.save(f"{train_sample_folder}/evaluation_labels", evaluation_labels)
    np.save(f"{train_sample_folder}/probability_labels", probability_labels)

    # Add old training samples
    for cur_path in old_training_samples:
        np.concatenate((sample_states, np.load(f"{cur_path}/game_states.npy")))
        np.concatenate((evaluation_labels, np.load(f"{cur_path}/evaluation_labels.npy")))
        np.concatenate((probability_labels, np.load(f"{cur_path}/probability_labels.npy")))

    counter = 0
    for x in sample_states:
        if np.amin(x) < 0 or np.amax(x) > 1:
            print(x)
            counter += 1
            if counter > 20:
                break
    for x in probability_labels:
        if np.amin(x) < 0 or np.amax(x) > 1:
            print(x)
            counter += 1
            if counter > 40:
                break
    for x in evaluation_labels:
        if x < -1 or x > 1:
            print(x)
            counter += 1
            if counter > 60:
                break
    # Load training model
    training_model = load_model(cur_gen_location)
    # Set learning rate
    K.set_value(training_model.optimizer.learning_rate, learning_rate)
    # Train neural net
    training_model.fit(
        x=sample_states,
        y=[evaluation_labels, probability_labels],
        batch_size=batch_size,
        epochs=epochs,
        shuffle=True,
    )

    # Save model
    training_model.save(new_model_location)

    # Do we want to save this time?
    print(f"Training took {format_time(time.perf_counter() - start_time)}!")

    # Testing

    cdef Trainer* tester = new Trainer(
        num_test_games,
        num_logged,
        iterations,
        c_puct,
        epsilon,
        test_log_folder.encode(),
        rng.integers(65536),  # Random seed
        True,  # Testing
    )

    cdef np.ndarray[np.float32_t, ndim=1] evaluations_1 = np.zeros(num_test_games, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] evaluations_2 = np.zeros(num_test_games, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] probabilities_1 = np.zeros((num_test_games, NUM_TOTAL_MOVES), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] probabilities_2 = np.zeros((num_test_games, NUM_TOTAL_MOVES), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] test_dirichlet = np.zeros(num_test_games*NUM_MOVES, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] test_game_states = np.zeros((num_test_games, GAME_STATE_SIZE), dtype=np.float32)

    print(f"Begin testing!")

    start_time = time.perf_counter()

    # First iteration
    tester.do_iteration(&evaluations_1[0], &probabilities_1[0,0], &test_dirichlet[0], &test_game_states[0,0])

    while True:

        res = training_model.predict(
            x=game_states, batch_size=num_test_games, verbose=0
        )
        evaluations_1 = res[0].flatten()
        probabilities_1 = res[1]
        res = model.predict(
            x=test_game_states, batch_size=num_test_games, verbose=0
        )
        evaluations_2 = res[0].flatten()
        probabilities_2 = res[1]

        test_dirichlet = rng.dirichlet((0.3,), num_games*NUM_MOVES).reshape((num_games*NUM_MOVES,)).astype(np.float32)

        res = tester.do_iteration(&evaluations_1[0], &probabilities_1[0,0],
            &evaluations_2[0], &probabilities_2[0,0],
            &test_dirichlet[0], &test_game_states[0,0],
        )

        if res:
            break

    print(f"Testing took {format_time(time.perf_counter() - start_time)}!")

    # Clear old models
    # Do we need to do this?
    keras.backend.clear_session()

    score = tester.get_score()

    del tester

    print(f"New agent score {score:1f}!")

    if score > testing_threshold:
        return True

    return False


