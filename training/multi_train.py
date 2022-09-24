import os
import sys
import time
import datetime
import numpy as np
import random
import argparse
from multiprocessing import Pool

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import keras.api._v2.keras as keras
from keras.api._v2.keras import Input, regularizers
from keras.api._v2.keras.models import Model
from keras.api._v2.keras.layers import Activation, Dense, BatchNormalization
from keras.api._v2.keras.optimizers import Adam
from trainer import Trainer
from tester import Tester
from util import *


if __name__ == "__main__":

    # Parse flags
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_games",
        type=int,
        default=3000,
        help="Number of games used for training a generation",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=200,
        help="Number of iterations/searches per turn",
    )
    parser.add_argument(
        "--num_test_games",
        type=int,
        default=400,
        help="Number of games used to test a generation. Default 400.",
    )
    parser.add_argument(
        "--series_length",
        type=int,
        default=1,
        help="Number of games played per SelfPlayer object. Default 1.",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=1,
        help="Number of processes used for self play. Default 1.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2048,
        help="Batch size for neural network training. Default 2048.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of epochs for neural network training. Default 1.",
    )
    parser.add_argument(
        "--testing",
        type=bool,
        default=False,
        help="Flag for testing that the code works. Uses small number for all arguments to run a generation quickly. Default False",
    )
    parser.add_argument(
        "--profiling",
        type=bool,
        default=False,
        help="Flag to toggle profiling for the first generation. Default is False",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="",
        help="Name of the run. Will write outputs into folder train_{name} or continue a previous run if the folder exists."
        "Default is the empty string, in which case timestamp is used",
    )

    # Dictionary of flag values
    args = vars(parser.parse_args())

    NUM_GAMES = max(1, args["num_games"])
    ITERATIONS = max(2, args["iterations"])
    NUM_TEST_GAMES = max(2, 2 * (args["num_test_games"] // 2)) # Enforce even number (first player bias)
    SERIES_LENGTH = max(1, args["series_length"])
    PROCESSES = max(1, args["processes"])
    PROFILING = args["profiling"]
    BATCH_SIZE = args["batch_size"]
    EPOCHS = max(1, args["epochs"])
    NAME = args["name"]

    # Testing, override arguments
    if args["testing"]:
        NUM_GAMES = 10
        ITERATIONS = 5
        NUM_TEST_GAMES = 10
        SERIES_LENGTH = 1
        NAME = "test"

    # "Normalize" position where we create files, etc.
    cwd = f"{os.getcwd()}/training"

    if not os.path.isdir(cwd):
        os.mkdir(cwd)

    while True:

        # Initialize training and playing models
        # Start from the previous loop or a previous run
        if len(NAME) > 0 and os.path.isdir(os.path.join(cwd, f"train_{NAME}")):

            current_generation = int(
                open(
                    f"{cwd}/train_{NAME}/metadata/current_generation.txt",
                    encoding="utf-8",
                )
                .read()
                .strip()
            )
            best_generation = int(
                open(
                    f"{cwd}/train_{NAME}/metadata/best_generation.txt", encoding="utf-8"
                )
                .read()
                .strip()
            )

            training_model = keras.models.load_model(
                f"{cwd}/train_{NAME}/generations/gen_{current_generation}/model"
            )

        # Initialize new "run"
        else:

            # Generate name from timestamp
            if len(NAME) == 0:
                now = datetime.datetime.now()
                NAME = (
                    f"{now.year}{now.month}{now.day}{now.hour}{now.minute}{now.second}"
                )

            # Create folder, implausible that name is repeated
            os.mkdir(f"{cwd}/train_{NAME}")

            # Create model
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

            os.mkdir(f"{cwd}/train_{NAME}/generations")
            os.mkdir(f"{cwd}/train_{NAME}/generations/gen_0")
            model.save(f"{cwd}/train_{NAME}/generations/gen_0/model")

            os.mkdir(f"{cwd}/train_{NAME}/metadata")
            open(
                f"{cwd}/train_{NAME}/metadata/current_generation.txt",
                "w+",
                encoding="utf-8",
            ).write("0")
            open(
                f"{cwd}/train_{NAME}/metadata/best_generation.txt",
                "w+",
                encoding="utf-8",
            ).write("0")
            current_generation = 0
            best_generation = 0

            training_model = keras.models.load_model(
                f"{cwd}/train_{NAME}/generations/gen_0/model"
            )

        # Training

        print(
            f"Began training generation {current_generation+1}!\nTraining with generation {best_generation}."
        )

        # Prepare directories
        try:
            os.mkdir(f"{cwd}/train_{NAME}/generations/gen_{current_generation+1}")
            os.mkdir(
                f"{cwd}/train_{NAME}/generations/gen_{current_generation+1}/metadata"
            )
            os.mkdir(f"{cwd}/train_{NAME}/generations/gen_{current_generation+1}/logs")
            os.mkdir(
                f"{cwd}/train_{NAME}/generations/gen_{current_generation+1}/logs/training_games"
            )
            os.mkdir(
                f"{cwd}/train_{NAME}/generations/gen_{current_generation+1}/logs/testing_games"
            )
        except FileExistsError:
            print(
                f"{cwd}/train_{NAME}/generations/gen_{current_generation+1} already exists"
            )

        open(
            f"{cwd}/train_{NAME}/generations/gen_{current_generation+1}/metadata/metadata.txt",
            "w+",
            encoding="utf-8",
        ).write(
            f"Playing model: {best_generation}\n"
            f"Number of games: {NUM_GAMES}\n"
            f"Number of searches: {ITERATIONS}\n"
            f"Number of test games: {NUM_TEST_GAMES}\n"
            f"Training sample batch size: {BATCH_SIZE}\n"
            f"Number of training epochs: {EPOCHS}\n"
            f"Number of processes used: {PROCESSES}\n"
        )

        res = []

        # We want to use multiprocessing
        if PROCESSES > 1:

            trainers = []
            logging = True

            # Set up trainers
            for _ in range(PROCESSES):
                trainers.append(
                    Trainer(
                        model_path=f"{cwd}/train_{NAME}/generations/gen_{best_generation}/model",
                        logging_path=f"{cwd}/train_{NAME}/generations/gen_{current_generation+1}/logs/training_games",
                        num_games=max(1, NUM_GAMES // PROCESSES),
                        iterations=ITERATIONS,
                        series_length=SERIES_LENGTH,
                        logging=logging,
                    )
                )
                logging = False

            start_time = time.time()

            # Run training. This is the meat of the process
            pool = Pool(processes=PROCESSES)
            res = pool.map(helper, trainers)
            pool.close()

        else:
            trainer = Trainer(
                        model_path=f"{cwd}/train_{NAME}/generations/gen_{best_generation}/model",
                        logging_path=f"{cwd}/train_{NAME}/generations/gen_{current_generation+1}/logs/training_games",
                        num_games=NUM_GAMES,
                        iterations=ITERATIONS,
                        series_length=SERIES_LENGTH,
                        logging=True,
                    )

            start_time = time.time()

            res = (trainer.play(),)


        open(
            f"{cwd}/train_{NAME}/generations/gen_{current_generation+1}/metadata/metadata.txt",
            "a",
            encoding="utf-8",
        ).write(f"Time to play training games: {format_time(time.time()-start_time)}\n")

        # Collect training samples
        samples = []
        eval_labels = []
        prob_labels = []
        for item in res:
            samples.extend(item[0])
            eval_labels.extend(item[1])
            prob_labels.extend(item[2])

        # Train neural nets
        training_model.fit(
            x=np.array(samples),
            y=[
                np.array(eval_labels),
                np.array(prob_labels),
            ],
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            shuffle=True,
        )

        # Save newly trained model
        training_model.save(
            f"{cwd}/train_{NAME}/generations/gen_{current_generation+1}/model"
        )

        print(
            f"Began testing generation {current_generation+1}!\nPlaying against generation {best_generation}."
        )

        # Testing

        res = []

        if PROCESSES > 1:

            testers = []
            logging = True

            for _ in range(PROCESSES):
                testers.append(
                    Tester(
                        model_1_path=f"{cwd}/train_{NAME}/generations/gen_{current_generation+1}/model",
                        model_2_path=f"{cwd}/train_{NAME}/generations/gen_{best_generation}/model",
                        logging_path=f"{cwd}/train_{NAME}/generations/gen_{current_generation+1}/logs/testing_games",
                        num_games=max(1, NUM_TEST_GAMES // PROCESSES),
                        iterations=ITERATIONS,
                        logging=logging,
                    )
                )
                logging = False

            start_time = time.time()

            pool = Pool(processes=PROCESSES)
            res = pool.map(helper, testers)
            pool.close()

        else:
            tester = Tester(
                        model_1_path=f"{cwd}/train_{NAME}/generations/gen_{current_generation+1}/model",
                        model_2_path=f"{cwd}/train_{NAME}/generations/gen_{best_generation}/model",
                        logging_path=f"{cwd}/train_{NAME}/generations/gen_{current_generation+1}/logs/testing_games",
                        num_games=NUM_GAMES,
                        iterations=ITERATIONS,
                        logging=True,
                    )

            start_time = time.time()

            res = (tester.play(),)

        open(
            f"{cwd}/train_{NAME}/generations/gen_{current_generation+1}/metadata/metadata.txt",
            "a",
            encoding="utf-8",
        ).write(f"Time to play testing games: {format_time(time.time()-start_time)}\n")

        score = sum(res) / (PROCESSES * (max(1, NUM_TEST_GAMES // PROCESSES)))

        open(
            f"{cwd}/train_{NAME}/generations/gen_{current_generation+1}/metadata/metadata.txt",
            "a",
            encoding="utf-8",
        ).write(f"Testing score: {score}\n")

        # New neural net scores >50%, update best_generation
        if score > 0.5:
            best_generation = current_generation + 1
        current_generation += 1

        # Update model generations used
        open(
            f"{cwd}/train_{NAME}/metadata/current_generation.txt", "w", encoding="utf-8"
        ).write(str(current_generation))
        open(
            f"{cwd}/train_{NAME}/metadata/best_generation.txt", "w", encoding="utf-8"
        ).write(str(best_generation))

        # Clear old models
        keras.backend.clear_session()
