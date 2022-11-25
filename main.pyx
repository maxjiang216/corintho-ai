# distutils: language = c++

from libcpp.string cimport string
from libcpp cimport bool
import numpy as np
cimport numpy as np
import time

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

def do():
    start_time = time.perf_counter()
    num_games = 3000        
    cdef Trainer* x = new Trainer(num_games, 1, 200, 1, 0.25, "test", 0)
    cdef np.ndarray[np.float32_t, ndim=1] evaluations = np.zeros(num_games, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] probabilities = np.zeros((num_games, 96), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] dirichlet = np.zeros((num_games, 96), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] game_states = np.zeros((num_games, 70), dtype=np.float32)
    counter = 0
    while True:
        np.random.random_sample()
        for i in range(num_games):
            evaluations[i] = np.random.random_sample() * 2 - 1
            probabilities[i] = np.random.random_sample(96)
            probabilities[i] /= sum(probabilities[i])
            dirichlet[i] = np.random.dirichlet((0.3,), 96).reshape((96,))
        res = x.do_iteration(&evaluations[0], &probabilities[0,0], &dirichlet[0,0], &game_states[0,0])
        if res:
            break
        counter += 1
        if counter % 1000 == 0:
            print(f"{counter} iterations done!")
    print(f"Took {time.perf_counter() - start_time} seconds!")