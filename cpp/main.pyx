# distutils: language = c++

from libcpp.string cimport string
from libcpp cimport bool
import numpy as np

cdef extern from "trainer.cpp":
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
        cdef Trainer* x = new Trainer(1, 1, 200, 1, 0.25, "test", 0)
        cdef float[::1] evaluations = np.zeros(1, dtype=np.float32)
        cdef float[::1] probabilities = np.zeros(96, dtype=np.float32)
        cdef float[::1] dirichlet = np.zeros(96, dtype=np.float32)
        cdef float[::1] game_states = np.zeros(70, dtype=np.float32)
        x.do_iteration(&evaluations[0], &probabilities[0], &dirichlet[0], &game_states[0])
        print(game_states)