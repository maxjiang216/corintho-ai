import time
import multiprocessing as mp
import numpy as np


class NeuralNet:
    """
    Neural Net for evaluating Corintho positions
    """

    def __init__(self, model, simulator, batch_size=16):
        self.model = model
        self.simulator = simulator
        self.batch_size = batch_size
        self.samples = []
        self.labels = []

    @staticmethod
    def play_games(args):
        """
        array -> (array, array)
        args is in the format [simulator, num_games]
        simulator is the Simulator used to play games
        num_games is the number of games to play
        batch_num is a unique identifier for logging purposes
        """
        simulator = args[0]
        num_games = args[1]
        batch_num = args[2]
        samples = []
        labels = []
        legal_move_nums = []
        for i in range(num_games):
            # temp_samples, temp_labels, legal_move_num = simulator.play_game()
            # samples.extend(temp_samples)
            # labels.extend(temp_labels)
            # legal_move_nums.extend(legal_move_num)
            simulator.play_game()
            if num_games > 10 and i > 0 and i % (num_games // 10) == 0:
                print(f"{i/num_games*100:.1f}% completed batch {batch_num}")
        print(f"Complete batch {batch_num}!")
        return (samples, labels, legal_move_nums)

    def train_generation(self, num_games=100, concurrency=10, num_epochs=1):
        """
        int ->
        Generating training samples and fits model
        Players num_games games to generate samples
        """

        results = []
        args = [
            (self.simulator, num_games // concurrency, i) for i in range(concurrency)
        ]
        start_time = time.time()
        with mp.Pool(concurrency) as pool:
            results = pool.map(self.play_games, args)
            pool.close()
        time_taken = time.time() - start_time
        # legal_move_nums = []
        # for game in results:
        #    self.samples.extend(game[0])
        #    self.labels.extend(game[1])
        #    legal_move_nums.extend(game[2])
        print(f"Total time: {time_taken}")
        print(
            f"Average time per game: {(time.time() - start_time) / ((num_games // concurrency) * concurrency):.3f}s"
        )
        print(f"Total moves: {len(self.samples)}")
        print(
            f"Average moves per game: {len(self.samples)/((num_games // concurrency) * concurrency)}"
        )
        print(f"Average num legal moves: {np.mean(legal_move_nums)}")
        print(f"Average time per move: {time_taken / len(self.samples)}")

        self.model.fit(
            x=np.array(self.samples),
            y=np.array(self.labels),
            batch_size=self.batch_size,
            epochs=num_epochs,
            shuffle=True,
        )

    def predict(self, game):
        """
        Game -> float
        Gives evaluation of game
        """
        return self.model.predict(x=np.array([game.get_vector()]))

    def check(self, path="./models/check.txt"):
        """
        (str) ->
        Writes predictions for current samples compared to actual result
        Used as surface level check for reasonableness of trained neural net
        """
        check_file = open(path, "w", encoding="utf-8")
        start_time = time.time()
        # predictions = self.model.predict(x=np.array(self.samples), verbose=0)
        predictions = self.model.predict(x=np.array(self.samples), verbose=0)
        # if len(self.samples) > 10 and i > 0 and i % (len(self.samples) // 10) == 0:
        #    print(i / len(self.samples))
        time_taken = time.time() - start_time
        print(f"Seconds per prediction: {time_taken/len(self.samples)}")
        # for i, sample in enumerate(self.samples):
        #    check_file.write(f"{predictions[i]} {self.labels[i]} {sample}\n")
