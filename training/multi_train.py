import os
import shutil
import sys
import time
import numpy as np
import random
from multiprocessing import Pool

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import keras.api._v2.keras as keras
from keras.api._v2.keras import Input, regularizers
from keras.api._v2.keras.models import Model
from keras.api._v2.keras.layers import Activation, Dense, BatchNormalization
from keras.api._v2.keras.optimizers import Adam
from trainer import Trainer
from tester import Tester

NUM_GAMES = 3000
ITERATIONS = 200
NUM_TEST_GAMES = 400
SERIES_LENGTH = 2
BATCH_SIZE = 2048
EPOCHS = 1
PROCESSES = 5


def helper(player):
    """Helper function for training and testing"""
    return player.play()


def format_time(t):
    """Format string
    t is time in seconds"""

    if t < 60:
        return f"{t:.1f} seconds"
    if t < 3600:
        return f"{t/60:.1f} minutes"
    return f"{t/60/60:.1f} hours"


if __name__ == "__main__":

    cwd = f"{os.getcwd()}/training"

    if not os.path.isdir(cwd):
        os.mkdir(cwd)

    seed = ""
    if len(sys.argv) >= 2 and os.path.isdir(f"{cwd}/train_{sys.argv[1]}"):
        seed = sys.argv[1]

    while True:

        # Initialize training and playing models
        # Start from the previous loop or a previous run
        if len(seed) > 0:

            current_generation = int(
                open(
                    f"{cwd}/train_{seed}/metadata/current_generation.txt",
                    encoding="utf-8",
                )
                .read()
                .strip()
            )
            best_generation = int(
                open(
                    f"{cwd}/train_{seed}/metadata/best_generation.txt", encoding="utf-8"
                )
                .read()
                .strip()
            )

            training_model = keras.models.load_model(
                f"{cwd}/train_{seed}/generations/gen_{current_generation}/model"
            )

        # Initialize new "run"
        else:

            seed = f"{random.randrange(100000000):08}"

            # Create folder, implausible that name is repeated
            os.mkdir(f"{cwd}/train_{seed}")

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

            os.mkdir(f"{cwd}/train_{seed}/generations")
            os.mkdir(f"{cwd}/train_{seed}/generations/gen_0")
            model.save(f"{cwd}/train_{seed}/generations/gen_0/model")

            os.mkdir(f"{cwd}/train_{seed}/metadata")
            open(
                f"{cwd}/train_{seed}/metadata/current_generation.txt",
                "w+",
                encoding="utf-8",
            ).write("0")
            open(
                f"{cwd}/train_{seed}/metadata/best_generation.txt",
                "w+",
                encoding="utf-8",
            ).write("0")
            current_generation = 0
            best_generation = 0

            training_model = keras.models.load_model(
                f"{cwd}/train_{seed}/generations/gen_0/model"
            )

        # Training

        print(
            f"Began training generation {current_generation+1}!\nTraining with generation {best_generation}."
        )

        # Prepare directories
        os.mkdir(f"{cwd}/train_{seed}/generations/gen_{current_generation+1}")
        os.mkdir(f"{cwd}/train_{seed}/generations/gen_{current_generation+1}/metadata")
        os.mkdir(f"{cwd}/train_{seed}/generations/gen_{current_generation+1}/logs")
        os.mkdir(
            f"{cwd}/train_{seed}/generations/gen_{current_generation+1}/logs/training_games"
        )
        os.mkdir(
            f"{cwd}/train_{seed}/generations/gen_{current_generation+1}/logs/testing_games"
        )

        open(
            f"{cwd}/train_{seed}/generations/gen_{current_generation+1}/metadata/metadata.txt",
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

        trainers = []
        logging = True

        # Set up trainers
        for _ in range(PROCESSES):
            trainers.append(
                Trainer(
                    model_path=f"{cwd}/train_{seed}/generations/gen_{best_generation}/model",
                    logging_path=f"{cwd}/train_{seed}/generations/gen_{current_generation+1}/logs/training_games",
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

        open(
            f"{cwd}/train_{seed}/generations/gen_{current_generation+1}/metadata/metadata.txt",
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
            f"{cwd}/train_{seed}/generations/gen_{current_generation+1}/model"
        )

        print(
            f"Began testing generation {current_generation+1}!\nPlaying against generation {best_generation}."
        )

        # Testing

        testers = []
        logging = True

        for _ in range(PROCESSES):
            testers.append(
                Tester(
                    model_1_path=f"{cwd}/train_{seed}/generations/gen_{current_generation+1}/model",
                    model_2_path=f"{cwd}/train_{seed}/generations/gen_{best_generation}/model",
                    logging_path=f"{cwd}/train_{seed}/generations/gen_{current_generation+1}/logs/testing_games",
                    num_games=max(1, NUM_TEST_GAMES // PROCESSES),
                    iterations=ITERATIONS,
                    series_length=SERIES_LENGTH,
                    logging=logging,
                )
            )
            logging = False

        start_time = time.time()

        pool = Pool(processes=PROCESSES)
        res = pool.map(helper, testers)
        pool.close()

        open(
            f"{cwd}/train_{seed}/generations/gen_{current_generation+1}/metadata/metadata.txt",
            "a",
            encoding="utf-8",
        ).write(f"Time to play testing games: {format_time(time.time()-start_time)}\n")

        score = sum(res) / (PROCESSES * (max(1, NUM_TEST_GAMES // PROCESSES)))

        open(
            f"{cwd}/train_{seed}/generations/gen_{current_generation+1}/metadata/metadata.txt",
            "a",
            encoding="utf-8",
        ).write(f"Testing score: {score}\n")

        # New neural net scores >50%, update best_generation
        if score > 0.5:
            best_generation = current_generation + 1
        current_generation += 1

        # Update model generations used
        open(
            f"{cwd}/train_{seed}/metadata/current_generation.txt", "w", encoding="utf-8"
        ).write(str(current_generation))
        open(
            f"{cwd}/train_{seed}/metadata/best_generation.txt", "w", encoding="utf-8"
        ).write(str(best_generation))

        # Clear old models
        keras.backend.clear_session()
