from montecarlo import MonteCarlo
from simulator import Simulator
from player import RandomPlayer, HumanPlayer
from mcplayer import MonteCarloPlayer

sum = 0
for i in range(10):
    elos = [MonteCarloPlayer(100), RandomPlayer()]
    player1 = elos[i % 2]
    player2 = elos[1 - i % 2]
    sim = Simulator(player1, player2)
    sum += (-1) ** (i % 2) * sim.play_game()

print(sum)
