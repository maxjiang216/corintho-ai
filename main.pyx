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
        int count_samples();
        void write_samples(float *game_states, float *evaluation_samples, float *probability_samples);
        float get_score();

NUM_TOTAL_MOVES = 96
NUM_MOVES = 96
GAME_STATE_SIZE = 70

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
    num_games=25000,
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

    # Training

    # Load playing model
    model = load_model(best_gen_location)

    cdef Trainer* trainer = new Trainer(
        num_games,
        num_logged,
        iterations,
        c_puct,
        epsilon,
        train_log_folder.encode(),
        0,  # Random seed
    )

    cdef np.ndarray[np.float32_t, ndim=1] evaluations = np.zeros(num_games, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] probabilities = np.zeros((num_games, NUM_TOTAL_MOVES), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] dirichlet = np.zeros((num_games, NUM_MOVES), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] game_states = np.zeros((num_games, GAME_STATE_SIZE), dtype=np.float32)

    evaluations_done = 0

    print(f"Begin training!")

    start_time = time.perf_counter()

    # First iteration
    trainer.do_iteration(&evaluations[0], &probabilities[0,0], &dirichlet[0,0], &game_states[0,0])

    while True:

        res = model.predict(
            x=game_states, batch_size=num_games, verbose=0
        )
        evaluations = res[0].flatten()
        probabiltiies = res[1]

        for i in range(num_games):
            dirichlet[i] = np.random.dirichlet((0.3,), 96).reshape((96,))
        res = trainer.do_iteration(&evaluations[0], &probabilities[0,0], &dirichlet[0,0], &game_states[0,0])
        if res:
            break
        evaluations_done += 1
        if evaluations_done % max(1, 15 * iterations // 100) == 0:
            time_taken = time.perf_counter() - start_time
            print(
                f"{evaluations_done} evaluations completed in {format_time(time_taken)}\n"
                f"Predicted time to complete: {format_time(26.67*iterations*time_taken/evaluations_done)}\n"
                f"Estimated time left: {format_time((26.67*iterations-evaluations_done)*time_taken/evaluations_done)}\n"
            )

    # Get training samples
    num_samples = trainer.count_samples()
    cdef np.ndarray[np.float32_t, ndim=2] sample_states = np.zeros((num_samples, GAME_STATE_SIZE), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] evaluation_labels = np.zeros(num_samples, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] probability_labels = np.zeros((num_samples, NUM_TOTAL_MOVES), dtype=np.float32)
    trainer.write_samples(&sample_states[0,0], &evaluation_labels[0], &probability_labels[0,0])

    # Save training samples
    np.save(f"{train_sample_folder}/game_states", sample_states)
    np.save(f"{train_sample_folder}/evaluation_labels", evaluation_labels)
    np.save(f"{train_sample_folder}/probability_labels", probability_labels)

    # Add old training samples
    for cur_path in old_training_samples:
        np.concatenate((sample_states, np.load(f"{cur_path}/game_states.npy")))
        np.concatenate((evaluation_labels, np.load(f"{cur_path}/evaluation_labels.npy")))
        np.concatenate((probability_labels, np.load(f"{cur_path}/probability_labels.npy")))

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
        test_log_folder,
        0,  # Random seed
    )

    cdef np.ndarray[np.float32_t, ndim=1] evaluations_1 = np.zeros(num_test_games, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] evaluations_2 = np.zeros(num_test_games, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] probabilities_1 = np.zeros((num_test_games, NUM_TOTAL_MOVES), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] probabilities_2 = np.zeros((num_test_games, NUM_TOTAL_MOVES), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] test_dirichlet = np.zeros((num_test_games, NUM_MOVES), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] test_game_states = np.zeros((num_test_games, GAME_STATE_SIZE), dtype=np.float32)

    print(f"Begin testing!")

    start_time = time.perf_counter()

    # First iteration
    tester.do_iteration(&evaluations_1[0], &probabilities_1[0,0], &test_dirichlet[0,0], &test_game_states[0,0])

    while True:

        evaluations_2, probabilities_2 = training_model.predict(
            x=game_states, batch_size=num_test_games, verbose=0
        )
        evaluations_1, probabilities_1 = model.predict(
            x=game_states, batch_size=num_test_games, verbose=0
        )

        for i in range(num_games):
            dirichlet[i] = np.random.dirichlet((0.3,), 96).reshape((96,))

        res = tester.do_iteration(&evaluations_1[0], &evaluations_1[0],
            &probabilities_1[0,0], &probabilities_2[0,0],
            &test_dirichlet[0,0], &test_game_states[0,0],
        )

        if res:
            break

    print(f"Testing took {format_time(time.perf_counter() - start_time)}!")

    # Clear old models
    # Do we need to do this?
    keras.backend.clear_session()

    score = tester.get_score()

    if score > testing_threshold:
        return True

    return False


