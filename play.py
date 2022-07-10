from simulator import Simulator
from mcplayer import MonteCarloPlayer
from player import RandomPlayer, HumanPlayer

simulator = Simulator(HumanPlayer(), MonteCarloPlayer(2400))
simulator.play_game()
