import numpy as np
import time
from selfplayer import SelfPlayer
from keras.api._v2.keras.models import load_model


def format_time(t):
    """Format string
    t is time in seconds"""

    if t < 60:
        return f"{t:.1f} seconds"
    if t < 3600:
        return f"{t/60:.1f} minutes"
    return f"{t/60/60:.1f} hours"


class Tester:
    """
    Test models
    """

    def __init__(
        self,
        model_1_path,
        model_2_path,
        logging_path,
        num_games=400,
        iterations=200,
        logging=False,
    ):
        """
        (int) -> Trainer
        Will train with num_games games concurrently
        """

        self.games = []
        self.model_1 = None
        self.model_2 = None
        self.model_1_path = model_1_path
        self.model_2_path = model_2_path
        self.logging_path = logging_path
        self.num_games = num_games
        self.iterations = iterations
        self.logging = logging
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
        Play all num_games games
        """

        # Do these at evaluation to avoid pickling a large amount of data
        self.model_1 = load_model(self.model_1_path)
        self.model_2 = load_model(self.model_2_path)

        for i in range(self.num_games):
            self.games.append(
                SelfPlayer(iterations=self.iterations, test=True, seed=i % 2)
            )

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
            res1 = self.model_1.predict(
                x=np.array(positions), batch_size=len(positions), verbose=0
            )
            res2 = self.model_2.predict(
                x=np.array(positions), batch_size=len(positions), verbose=0
            )
            evaluations = list(
                zip(list(zip(res1[0], res1[1])), list(zip(res2[0], res2[1])))
            )
            if self.logging:
                evaluations_done += 1
                if evaluations_done % max(1, 15 * self.iterations // 100) == 0:
                    time_taken = time.time() - start_time
                    open(
                        f"{self.logging_path}/progress.txt",
                        "a",
                        encoding="utf-8",
                    ).write(
                        f"{evaluations_done} evaluations completed in {format_time(time_taken)}\n"
                        f"Predicted time to complete: {format_time(26.67*self.iterations*time_taken/evaluations_done)}\n"
                        f"Estimated time left: {format_time((26.67*self.iterations-evaluations_done)*time_taken/evaluations_done)}\n\n"
                    )
                    print(
                        f"{evaluations_done} evaluations completed in {format_time(time_taken)}\n"
                        f"Predicted time to complete: {format_time(26.67*self.iterations*time_taken/evaluations_done)}\n"
                        f"Estimated time left: {format_time((26.67*self.iterations-evaluations_done)*time_taken/evaluations_done)}\n"
                    )

        # Compile logs
        # Return game histories and outcome as string to be written into file
        game_logs_file = open(
            f"{self.logging_path}/game_logs.txt",
            "a",
            encoding="utf-8",
        )

        score = 0
        for i, game in enumerate(self.games):
            score += (game.game.outcome * (-1) ** game.seed + 1) / 2
            game_logs_file.write(
                f"GAME {i}\nRESULT: {game.game.outcome}\n"
                + "\n".join(game.logs)
                + "\n\n"
            )

        open(
            f"{self.logging_path}/game_stats.txt",
            "w+",
            encoding="utf-8",
        ).write(f"TIME TAKEN: {format_time(time.time()-start_time)}")

        # If testing, return win rate
        return score
