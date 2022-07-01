from montecarlo import MonteCarlo
from simulator import Simulator
from player import RandomPlayer, HumanPlayer
from mcplayer import MonteCarloPlayer
from board import Board
from game import Game
import time
import cProfile
import pstats

with cProfile.Profile() as pr:
    sum = 0
    for i in range(1):
        elos = [MonteCarloPlayer(600), MonteCarloPlayer(600)]
        player1 = elos[i % 2]
        player2 = elos[1 - i % 2]
        sim = Simulator(player1, player2)
        sum += (-1) ** (i % 2) * sim.play_game(True)

    print(sum)
    print(sim.samples)

stats = pstats.Stats(pr)
stats.print_stats()
