import numpy as np
import time
from selfplayer import SelfPlayer
from util import format_time
from keras.api._v2.keras.models import load_model


class Trainer:
    """
    Trains with multiple games
    Collects training samples
    """

    def __init__(
        self,
        model_path,
        logging_path,
        num_games=3000,
        iterations=200,
        series_length=1,
        logging=False,
        profiling=False,
    ):
        """
        (int) -> Trainer
        Will train with num_games games concurrently
        """

        self.games = []
        self.model = None
        self.model_path = model_path
        self.logging_path = logging_path
        self.iterations = iterations
        self.num_games = num_games
        self.series_length = series_length
        self.logging = logging
        self.profiling = profiling
        if logging:
            open(f"{logging_path}/progress.txt", "w+", encoding="utf-8",).write(
                f"{num_games} games with {iterations} searches per move\nStarted: {time.ctime()}\n\n"
            )
            open(
                f"{logging_path}/game_logs.txt",
                "w+",
                encoding="utf-8",
            ).write(f"{num_games} games with {iterations} searches per move\n\n")

    def play(self):
        """
        Play all num_games games and get training samples
        """

        # Do these at evaluation to avoid pickling a large amount of data
        self.model = load_model(self.model_path)

        for i in range(max(1, self.num_games // self.series_length)):
            self.games.append(
                SelfPlayer(
                    iterations=self.iterations,
                    series_length=self.series_length,
                    logging=i < 100 and self.logging,
                )
            )

        # Play games

        evaluations = [None] * len(self.games)
        evaluations_done = 0
        start_time = time.time()
        last_time = time.time()

        while True:

            res = []
            for i, game in enumerate(self.games):
                # Offset by delaying for the first few searches
                if i // self.iterations > evaluations_done:
                    break
                # Start of game not played in previous iteration
                if i >= len(evaluations):
                    res.append(game.play(None))
                else:
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
                x=np.array(positions), batch_size=len(positions), verbose=0
            )
            evaluations = list(zip(res[0], res[1]))
            if self.logging:
                evaluations_done += 1
                if evaluations_done % max(1, 15 * self.iterations // 100) == 0 or time.time() - last_time > 60:
                    time_taken = time.time() - start_time
                    last_time = time.time()
                    open(
                        f"{self.logging_path}/progress.txt",
                        "a",
                        encoding="utf-8",
                    ).write(
                        f"{evaluations_done} evaluations completed in {format_time(time_taken)}\n"
                        f"Predicted time to complete: {format_time(self.series_length*26.67*self.iterations*time_taken/evaluations_done)}\n"
                        f"Estimated time left: {format_time((self.series_length*26.67*self.iterations-evaluations_done)*time_taken/evaluations_done)}\n\n"
                    )
                    print(
                        f"{evaluations_done} evaluations completed in {format_time(time_taken)}\n"
                        f"Predicted time to complete: {format_time(self.series_length*26.67*self.iterations*time_taken/evaluations_done)}\n"
                        f"Estimated time left: {format_time((self.series_length*26.67*self.iterations-evaluations_done)*time_taken/evaluations_done)}\n"
                    )

        # Compile logs
        # Return game histories and outcome as string to be written into file
        game_logs_file = open(
            f"{self.logging_path}/game_logs.txt",
            "a",
            encoding="utf-8",
        )

        total_turns = []
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
            if len(game.logs) > 0:
                total_turns.append(len(game.logs) / 2)
                game_logs_file.write(
                    f"GAME {i}\nRESULT: {game.game.outcome}\n"
                    + "\n".join(game.logs)
                    + "\n\n"
                )

        open(f"{self.logging_path}/game_stats.txt", "w+", encoding="utf-8").write(
            f"AVERAGE NUMBER OF TURNS: {sum(total_turns) / len(total_turns):.2f}\n"
            f"TIME TAKEN: {format_time(time.time()-start_time)}"
        )

        return (samples, evaluation_labels, probability_labels)