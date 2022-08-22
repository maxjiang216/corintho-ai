import os
import shutil
from statistics import mode
import sys
import time
import numpy as np
from multiprocessing import Pool
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import keras.api._v2.keras as keras
from keras.api._v2.keras import Input, regularizers
from keras.api._v2.keras.models import Model
from keras.api._v2.keras.layers import Activation, Dense, BatchNormalization
from keras.api._v2.keras.optimizers import Adam
from selfplayer import SelfPlayer

NUM_GAMES = 3000
ITERATIONS = 200
NUM_TEST_GAMES = 400
BATCH_SIZE = 2048
EPOCHS = 1


def helper(game, evaluations):
    """Helper function"""
    return game.play(evaluations)


def format_time(t):
    """Format string
    t is time in seconds"""

    if t < 60:
        return f"{t:.1f} seconds"
    if t < 3600:
        return f"{t/60:.1f} minutes"
    return f"{t/60/60:.1f} hours"


if __name__ == "__main__":

    if len(sys.argv) < 2:
        input_layer = Input(shape=(70,))
        layer_1 = Activation("relu")(
            BatchNormalization()(
                Dense(
                    units=100,
                    kernel_regularizer=regularizers.L2(1e-4),
                )(input_layer)
            )
        )
        layer_2 = Activation("relu")(
            BatchNormalization()(
                Dense(
                    units=100,
                    kernel_regularizer=regularizers.L2(1e-4),
                )(layer_1)
            )
        )
        layer_3 = Activation("relu")(
            BatchNormalization()(
                Dense(
                    units=100,
                    kernel_regularizer=regularizers.L2(1e-4),
                )(layer_2)
            )
        )
        layer_4 = Activation("relu")(
            BatchNormalization()(
                Dense(
                    units=100,
                    kernel_regularizer=regularizers.L2(1e-4),
                )(layer_3)
            )
        )
        eval_layer = Activation("relu")(
            BatchNormalization()(
                Dense(
                    units=100,
                    kernel_regularizer=regularizers.L2(1e-4),
                )(layer_4)
            )
        )
        eval_output = Dense(units=1, activation="tanh")(eval_layer)
        prob_layer = Activation("relu")(
            BatchNormalization()(
                Dense(
                    units=100,
                    kernel_regularizer=regularizers.L2(1e-4),
                )(layer_4)
            )
        )
        prob_output = Dense(units=96, activation="softmax")(prob_layer)

        model = Model(inputs=input_layer, outputs=[eval_output, prob_output])
        model.compile(
            optimizer=Adam(learning_rate=0.01),
            loss=[
                keras.losses.MeanSquaredError(),
                keras.losses.CategoricalCrossentropy(),
            ],
        )

        playing_model = model
        training_model = model

        model_num = 1
    else:
        model_num = int(sys.argv[1])
        model = keras.models.load_model(f"./training/models/model_{model_num}")
        # Change this
        playing_model = model
        training_model = model

    fail_num = 0

    while True:

        start_time = time.time()

        print(f"Begin training generation {model_num}! (Times failed: {fail_num})")

        # Prepare log files
        try:
            shutil.rmtree(f"./training/models/model_{model_num}")
        except FileNotFoundError:
            pass
        os.mkdir(f"./training/models/model_{model_num}")
        os.mkdir(f"./training/models/model_{model_num}/logs")
        playing_model.save(f"./training/models/model_{model_num}/player_model")
        open(
            f"./training/models/model_{model_num}/logs/player_weights.txt",
            "w",
            encoding="utf-8",
        ).write(str(playing_model.get_weights()))
        open(
            f"./training/models/model_{model_num}/logs/untrained_weights.txt",
            "w",
            encoding="utf-8",
        ).write(str(training_model.get_weights()))
        open(
            f"./training/models/model_{model_num}/logs/training_game_progress.txt",
            "w",
            encoding="utf-8",
        ).write(
            f"{NUM_GAMES} games with {ITERATIONS} searches per move\nStarted: {start_time}\n\n"
        )
        open(
            f"./training/models/model_{model_num}/logs/training_logs.txt",
            "w",
            encoding="utf-8",
        ).write(
            f"{NUM_GAMES} games with {ITERATIONS} searches per move\nStarted: {start_time}\n\n"
        )

        # Training
        games = []
        for _ in range(NUM_GAMES):
            games.append(SelfPlayer(iterations=ITERATIONS))
        evaluations_done = 0
        evaluations = [None] * NUM_GAMES
        samples = []
        evaluation_labels = []
        probability_labels = []
        turns = 0

        with Pool(processes=4) as pool:

            while True:

                res = pool.starmap(helper, zip(games, evaluations))

                positions = []
                games = []
                for item in res:
                    # Game is done
                    if max(item[0]) > 0:
                        positions.append(item[0])
                        games.append(item[1])
                    else:
                        (
                            cur_samples,
                            cur_evaluation_labels,
                            cur_probability_labels,
                        ) = item[1].get_samples()
                        samples.extend(cur_samples)
                        evaluation_labels.extend(cur_evaluation_labels)
                        probability_labels.extend(cur_probability_labels)
                        turns += len(item[1].logs) / 2
                        training_logs_file = open(
                            f"./training/models/model_{model_num}/logs/training_logs.txt",
                            "a",
                            encoding="utf-8",
                        )
                        training_logs_file.write(f"RESULT: {item[1].game.outcome}\n")
                        training_logs_file.write("\n".join(item[1].logs))
                        training_logs_file.write("\n\n")
                        training_logs_file.close()
                # Done all games
                if len(games) == 0:
                    break
                res = model.predict(
                    x=np.array(positions), batch_size=len(positions), verbose=0
                )
                evaluations = list(zip(res[0], res[1]))
                evaluations_done += 1
                if evaluations_done % min(1, 15 * ITERATIONS // 100) == 0:
                    time_taken = time.time() - start_time
                    open(
                        f"./training/models/model_{model_num}/logs/training_game_progress.txt",
                        "a",
                        encoding="utf-8",
                    ).write(
                        f"{evaluations_done} evaluations completed in {format_time(time_taken)}\n"
                        f"Predicted time to complete: {format_time(26.67*ITERATIONS*time_taken/evaluations_done)}\n"
                        f"Predicted time left: {format_time((26.67*ITERATIONS-evaluations_done)*time_taken/evaluations_done)}\n\n"
                    )

        open(
            f"./training/models/model_{model_num}/logs/training_stats.txt",
            "w",
            encoding="utf-8",
        ).write(
            f"{NUM_GAMES} games with {ITERATIONS} searches per move played in {format_time(time.time()-start_time)}\n"
            f"AVERAGE MOVES PER GAME: {turns / NUM_GAMES}"
        )

        training_samples_file = open(
            f"./training/models/model_{model_num}/logs/training_samples.txt",
            "w",
            encoding="utf-8",
        )

        for sample, eval_label, prob_label in zip(
            samples, evaluation_labels, probability_labels
        ):
            training_samples_file.write(f"{sample}\n{eval_label}\n{prob_label}\n\n")

        training_samples_file.close()

        # Train neural nets
        training_model.fit(
            x=np.array(samples),
            y=[
                np.array(evaluation_labels),
                np.array(probability_labels),
            ],
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            shuffle=True,
        )

        training_model.save(f"./training/models/model_{model_num}/new_model")

        start_time = time.time()

        print(f"Begin testing generation {model_num}!")

        # Prepare log files
        open(
            f"./training/models/model_{model_num}/logs/trained_weights.txt",
            "w",
            encoding="utf-8",
        ).write(str(training_model.get_weights()))
        open(
            f"./training/models/model_{model_num}/logs/testing_game_progress.txt",
            "w",
            encoding="utf-8",
        ).write(
            f"{NUM_TEST_GAMES} games with {ITERATIONS} searches per move\nStarted: {start_time}\n\n"
        )
        open(
            f"./training/models/model_{model_num}/logs/testing_game_logs.txt",
            "w",
            encoding="utf-8",
        ).write(
            f"{NUM_TEST_GAMES} games with {ITERATIONS} searches per move\nStarted: {start_time}\n\n"
        )

        # Testing
        old_model = keras.models.load_model(
            f"./training/models/model_{model_num}/player_model"
        )
        games = []
        for i in range(NUM_TEST_GAMES):
            games.append(SelfPlayer(iterations=ITERATIONS, test=True, seed=i % 2))
        evaluations_done = 0
        evaluations = [None] * NUM_TEST_GAMES
        samples = []
        evaluation_labels = []
        probability_labels = []
        turns = 0
        # Score of first player (first model)
        score = 0

        with Pool(processes=4) as pool:

            while True:

                res = pool.starmap(helper, zip(games, evaluations))

                positions = []
                games = []
                for item in res:
                    # Game is done
                    if max(item[0]) > 0:
                        positions.append(item[0])
                        games.append(item[1])
                    else:
                        score += (item[1].game.outcome * (-1) ** item[1].seed + 1) / 2
                        turns += len(item[1].logs) / 2
                        testing_logs_file = open(
                            f"./training/models/model_{model_num}/logs/testing_game_logs.txt",
                            "a",
                            encoding="utf-8",
                        )
                        testing_logs_file.write(
                            f"SEED: {item[1].seed}\n"
                            f"RESULT: {item[1].game.outcome}\n"
                        )
                        testing_logs_file.write("\n".join(item[1].logs))
                        testing_logs_file.write("\n\n")
                        testing_logs_file.close()

                # Done all games
                if len(games) == 0:
                    break
                res1 = training_model.predict(
                    x=np.array(positions), batch_size=len(games), verbose=0
                )
                res2 = old_model.predict(
                    x=np.array(positions), batch_size=len(games), verbose=0
                )
                evaluations = list(
                    zip(list(zip(res1[0], res1[1])), list(zip(res2[0], res2[1])))
                )
                evaluations_done += 1
                if evaluations_done % min(1, 15 * ITERATIONS // 100) == 0:
                    time_taken = time.time() - start_time
                    open(
                        f"./training/models/model_{model_num}/logs/testing_game_progress.txt",
                        "a",
                        encoding="utf-8",
                    ).write(
                        f"{evaluations_done} evaluations completed in {format_time(time_taken)}\n"
                        f"Predicted time to complete: {format_time(26.67*ITERATIONS*time_taken/evaluations_done)}\n"
                        f"Predicted time left: {format_time((26.67*ITERATIONS-evaluations_done)*time_taken/evaluations_done)}\n\n"
                    )

        testing_game_progress_file.close()

        open(
            f"./training/models/model_{model_num}/logs/testing_stats.txt",
            "w",
            encoding="utf-8",
        ).write(
            f"FIRST MODEL SCORE: {score/NUM_TEST_GAMES}\nTIMES FAILED: {fail_num}"
            f"{NUM_TEST_GAMES} games with {ITERATIONS} searches per move played in {format_time(time.time()-start_time)}\n"
            f"AVERAGE MOVES PER GAME: {turns / NUM_TEST_GAMES}"
        )

        # New neural net scores >50%
        if score / NUM_TEST_GAMES > 0.5:
            playing_model = training_model
            fail_num = 0
        else:
            fail_num += 1

        model_num += 1
