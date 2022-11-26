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
from keras.api._v2.keras import Input, regularizers
from keras.api._v2.keras.models import Model
from keras.api._v2.keras.layers import Activation, Dense, BatchNormalization
from keras.api._v2.keras.optimizers import Adam
from keras.api._v2.keras.models import load_model

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

NUM_TOTAL_MOVES = 96
NUM_MOVES = 56
GAME_STATE_SIZE = 70

def train_generation(*,
    cur_gen_location,  # neural network to train on
    best_gen_location,  # opponent to test against
    train_log_folder,
    test_log_folder,
    train_sample_file,  # String path to write training samples into
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
    old_training_samples=[],  # list of files containing training samples from previous generations to use
):

    # Training

    # Load model
    model = load_model(cur_gen_location)

    cdef Trainer* trainer = new Trainer(
        num_games,
        num_logged,
        iterations,
        c_puct,
        epsilon,
        train_log_folder,
        0,  # Random seed
    )

    cdef np.ndarray[np.float32_t, ndim=1] evaluations = np.zeros(num_games, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] probabilities = np.zeros((num_games, NUM_TOTAL_MOVES), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] dirichlet = np.zeros((num_games, NUM_MOVES), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] game_states = np.zeros((num_games, GAME_STATE_SIZE), dtype=np.float32)

    counter = 0
    start_time = time.perf_counter()

    # First iteration
    trainer.do_iteration(&evaluations[0], &probabilities[0,0], &dirichlet[0,0], &game_states[0,0])

    while True:

        evaluations, probabilities = model.predict(
            x=game_states, batch_size=num_games, verbose=0
        )

        for i in range(num_games):
            dirichlet[i] = np.random.dirichlet((0.3,), 96).reshape((96,))
        res = trainer.do_iteration(&evaluations[0], &probabilities[0,0], &dirichlet[0,0], &game_states[0,0])
        if res:
            break
        counter += 1
        if counter % 1000 == 0:
            print(f"{counter} iterations done!")

    print(f"Took {time.perf_counter() - start_time} seconds!")
