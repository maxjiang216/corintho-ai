from simulator import Simulator
from player import RandomPlayer

player1 = RandomPlayer()
player2 = RandomPlayer()
sim = Simulator(player1, player2)
print(sim.play_game())
