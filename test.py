from simulator import Simulator
from player import RandomPlayer, HumanPlayer

player1 = RandomPlayer()
player2 = HumanPlayer()
sim = Simulator(player1, player2)
print(sim.play_game())
