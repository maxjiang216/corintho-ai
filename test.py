from simulator import Simulator
from player import RandomPlayer, HumanPlayer
from mcplayer import MonteCarloPlayer

sum = 0
for i in range(1):
    player1 = MonteCarloPlayer()
    player2 = HumanPlayer()
    sim = Simulator(player1, player2)
    sum += sim.play_game()

print(sum)
