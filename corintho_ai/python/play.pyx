# distutils: language = c++

import os
import time

# Set TensorFlow logging level (prevent imports from logging a lot of stuff)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import keras.api._v2.keras as keras
import numpy as np
from keras.api._v2.keras.models import load_model

cimport numpy as np
from libcpp cimport bool
from libcpp.string cimport string


cdef extern from "../cpp/playmc.cpp":
    cdef cppclass PlayMC:
        PlayMC()
        PlayMC(int num_iterations, int searches_per_eval,
               float c_puct, float epsilon, bool logging, int seed)
        bool do_iteration(float *evaluations, float *probabilities)
        int choose_move()
        void receive_opp_move(int move_choice)
        int write_requests(float *game_states)
        bool is_done()
        bool has_won()
        bool has_drawn()
        void print_game()

def encode_move(move_str):
    """Returns move ID correspond to inputted string"""

    move_dict = {
        "a4R": 0,
        "b4R": 1,
        "c4R": 2,
        "a3R": 3,
        "b3R": 4,
        "c3R": 5,
        "a2R": 6,
        "b2R": 7,
        "c2R": 8,
        "a1R": 9,
        "b1R": 10,
        "c1R": 11,
        "a4D": 12,
        "b4D": 13,
        "c4D": 14,
        "d4D": 15,
        "a3D": 16,
        "b3D": 17,
        "c3D": 18,
        "d3D": 19,
        "a2D": 20,
        "b2D": 21,
        "c2D": 22,
        "d2D": 23,
        "b4L": 24,
        "c4L": 25,
        "d4L": 26,
        "b3L": 27,
        "c3L": 28,
        "d3L": 29,
        "b2L": 30,
        "c2L": 31,
        "d2L": 32,
        "b1L": 33,
        "c1L": 34,
        "d1L": 35,
        "a3U": 36,
        "b3U": 37,
        "c3U": 38,
        "d3U": 39,
        "a2U": 40,
        "b2U": 41,
        "c2U": 42,
        "d2U": 43,
        "a1U": 44,
        "b1U": 45,
        "c1U": 46,
        "d1U": 47,
        "Ba4": 48,
        "Bb4": 49,
        "Bc4": 50,
        "Bd4": 51,
        "Ba3": 52,
        "Bb3": 53,
        "Bc3": 54,
        "Bd3": 55,
        "Ba2": 56,
        "Bb2": 57,
        "Bc2": 58,
        "Bd2": 59,
        "Ba1": 60,
        "Bb1": 61,
        "Bc1": 62,
        "Bd1": 63,
        "Ca4": 64,
        "Cb4": 65,
        "Cc4": 66,
        "Cd4": 67,
        "Ca3": 68,
        "Cb3": 69,
        "Cc3": 70,
        "Cd3": 71,
        "Ca2": 72,
        "Cb2": 73,
        "Cc2": 74,
        "Cd2": 75,
        "Ca1": 76,
        "Cb1": 77,
        "Cc1": 78,
        "Cd1": 79,
        "Aa4": 80,
        "Ab4": 81,
        "Ac4": 82,
        "Ad4": 83,
        "Aa3": 84,
        "Ab3": 85,
        "Ac3": 86,
        "Ad3": 87,
        "Aa2": 88,
        "Ab2": 89,
        "Ac2": 90,
        "Ad2": 91,
        "Aa1": 92,
        "Ab1": 93,
        "Ac1": 94,
        "Ad1": 95,
    }

    return move_dict[move_str]

def play(
    model_location,
    player_turn=0,
    iterations=51200,
    searches_per_eval=512,
    c_puct=1.0,
    epsilon=0.1,
    logging=False,
):

    cdef int NUM_MOVES = 96
    cdef int GAME_STATE_SIZE = 70

    rng = np.random.default_rng(int(time.time()))

    # Load playing model
    model = load_model(model_location)

    cdef PlayMC *cpu_player = new PlayMC(
        iterations,
        searches_per_eval,
        c_puct,
        epsilon,
        logging,
        rng.integers(65536),
    )

    cdef np.ndarray[np.float32_t, ndim=1] evaluations = np.zeros(searches_per_eval, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] probabilities = np.zeros((searches_per_eval, NUM_MOVES), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] game_states = np.zeros((searches_per_eval, GAME_STATE_SIZE), dtype=np.float32)

    eval = 0

    # Human player is first player
    if player_turn == 0:
        move = input("You are the first player. Please enter a move:\n")
        print(move, encode_move(move))
        cpu_player.receive_opp_move(encode_move(move))
        print("The new game position is:")
        cpu_player.print_game()

    # While the game is not done
    while not cpu_player.is_done():

        res = cpu_player.do_iteration(&evaluations[0], &probabilities[0,0])
        if res:  # Computer has completed searching for this turn
            move = cpu_player.choose_move()
            eval = 0
            print(f"CPU chooses move {move}.\nThe new game position is:")
            cpu_player.print_game()
            if not cpu_player.is_done():
                move = input("It is your turn. Please enter a move:\n")
                cpu_player.receive_opp_move(encode_move(move))
                print("The new game position is:")
                cpu_player.print_game()
        else:
            num_requests = cpu_player.write_requests(&game_states[0,0])
            if num_requests == 0:
                move = cpu_player.choose_move()
                eval = 0
                print(f"CPU chooses move {move}.\nThe new game position is:")
                cpu_player.print_game()
                if not cpu_player.is_done():
                    move = input("It is your turn. Please enter a move:\n")
                    cpu_player.receive_opp_move(encode_move(move))
                    print("The new game position is:")
                    cpu_player.print_game()
            else:
                print(f"eval {eval} {num_requests}")
                res = model.predict(
                    x=game_states, verbose=0,
                )
                evaluations = res[0].flatten()
                probabilities = res[1]
                eval += 1

    # End game stuff
    print("The game has completed")
    if cpu_player.has_won():
        print("The CPU player won!")
    elif cpu_player.has_drawn():
        print("You drew the CPU player!")
    else:
        print("You won!")

    del cpu_player