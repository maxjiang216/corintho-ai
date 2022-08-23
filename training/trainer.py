import numpy as np
import time
from selfplayer import SelfPlayer


def format_time(t):
    """Format string
    t is time in seconds"""

    if t < 60:
        return f"{t:.1f} seconds"
    if t < 3600:
        return f"{t/60:.1f} minutes"
    return f"{t/60/60:.1f} hours"


class Trainer:
    """
    Trains with multiple games
    Collects training samples
    """

    def __init__(self, model, model_num, num_games=3000, iterations=200, logging=False):
        """
        (int) -> Trainer
        Will train with num_games games concurrently
        """

        self.games = []
        self.model = model
        self.model_num = model_num
        self.iterations = iterations
        self.num_games = num_games
        self.logging = logging
        if logging:
            open(
                f"./training/models/model_{model_num}/logs/training_game_progress.txt",
                "w",
                encoding="utf-8",
            ).write(
                f"{num_games} games with {iterations} searches per move\nStarted: {time.ctime()}\n\n"
            )
            open(
                f"./training/models/model_{model_num}/logs/training_game_logs.txt",
                "w",
                encoding="utf-8",
            ).write(f"{num_games} games with {iterations} searches per move\n\n")

    def play(self):
        """
        Play all num_games games and get training samples
        """

        for _ in range(self.num_games):
            self.games.append(SelfPlayer(iterations=self.iterations))

        # Play games

        evaluations = [None] * len(self.games)
        evaluations_done = 0
        start_time = time.time()

        while True:

            res = []
            for i, game in enumerate(self.games):
                res.append(game.play(evaluations[i]))

            positions = []
            games_done = 0
            for item in res:
                # Game is done
                if max(item) == 0:
                    games_done += 1
                positions.append(item)
            # Done all games
            if games_done == len(self.games):
                break
            res = self.model.predict(
                x=np.array(positions), batch_size=len(self.games), verbose=0
            )
            evaluations = list(zip(res[0], res[1]))
            if self.logging:
                evaluations_done += 1
                print(evaluations_done)
                if evaluations_done % max(1, 15 * self.iterations // 100) == 0:
                    time_taken = time.time() - start_time
                    open(
                        f"./training/models/model_{self.model_num}/logs/training_game_progress.txt",
                        "a",
                        encoding="utf-8",
                    ).write(
                        f"{evaluations_done} evaluations completed in {format_time(time_taken)}"
                        f"Predicted time to complete: {format_time(26.67*self.iterations*time_taken/evaluations_done)}"
                        f"Estimated time left: {format_time((26.67*self.iterations-evaluations_done)*time_taken/evaluations_done)}"
                    )

        # Compile logs
        # Return game histories and outcome as string to be written into file
        game_logs_file = open(
            f"./training/models/model_{self.model_num}/logs/training_game_logs.txt",
            "a",
            encoding="utf-8",
        )
        total_turns = 0
        samples = []
        evaluation_labels = []
        probability_labels = []
        for i, game in enumerate(self.games):
            (
                cur_samples,
                cur_evaluation_labels,
                cur_probability_labels,
            ) = game.get_samples()
            samples.extend(cur_samples)
            evaluation_labels.extend(cur_evaluation_labels)
            probability_labels.extend(cur_probability_labels)
            total_turns += len(game.logs) / 2
            game_logs_file.write(f"GAME {i}\nRESULT: {game.game.outcome}\n").write(
                "\n".join(game.logs)
            ).write("\n\n")

        open(
            f"./training/models/model_{self.model_num}/logs/training_game_stats.txt",
            "w",
            encoding="utf-8",
        ).write(
            f"AVERAGE NUMBER OF TURNS: {total_turns / len(self.games):.2f}\n"
            f"TIME TAKEN: {format_time(time.time()-start_time)}"
        )

        # If testing, return win rate
        return (samples, evaluation_labels, probability_labels)
