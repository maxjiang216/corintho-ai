from montecarlo import MonteCarlo
from simulator import Simulator
from player import RandomPlayer, HumanPlayer
from mcplayer import MonteCarloPlayer
from board import Board
from game import Game
import time
import cProfile
import pstats

v = [1, 2, 3]
print(v[3:])

with cProfile.Profile() as pr:
    sum = 0
    t = time.time()
    for i in range(1):
        elos = [MonteCarloPlayer(600), MonteCarloPlayer(600)]
        player1 = elos[i % 2]
        player2 = elos[1 - i % 2]
        sim = Simulator(player1, player2)
        sum += (-1) ** (i % 2) * sim.play_game()

    print(sum)
    d = time.time() - t
    print(d)
    print(Board.GetLinesTime)
    print(Game.GetLegalMovesTime)
    print(Game.GetLegalMovesTime / d)
    print(Board.GetLinesTime / Game.GetLegalMovesTime)
    print(Board.CanMoveTime / Game.GetLegalMovesTime)
    print(Board.CanPlaceTime / Game.GetLegalMovesTime)
    print(Game.IsLegalTime / Game.GetLegalMovesTime)
    print(Game.DeepCopyTime / Game.GetLegalMovesTime)
    print(Game.DoMoveTime / Game.GetLegalMovesTime)

stats = pstats.Stats(pr)
stats.print_stats()
