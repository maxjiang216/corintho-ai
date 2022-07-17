from implement.simulator import Simulator
from mc.mcplayer import MonteCarloPlayer
from implement.player import RandomPlayer, HumanPlayer

for i in range(100):
    print(i)
    simulator = Simulator(MonteCarloPlayer(100), RandomPlayer())
    simulator.play_game()
