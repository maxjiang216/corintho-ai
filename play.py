from implement.simulator import Simulator
from mc.mcplayer import MonteCarloPlayer
from implement.player import RandomPlayer, HumanPlayer

simulator = Simulator(HumanPlayer(), RandomPlayer())
simulator.play_game()
