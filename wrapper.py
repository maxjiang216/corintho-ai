from corintho import train_generation
import argparse
import os
import shutil
import time
import datetime
from multiprocessing import cpu_count

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import keras.api._v2.keras as keras
from keras.api._v2.keras import Input, regularizers
from keras.api._v2.keras.models import Model
from keras.api._v2.keras.layers import Activation, Dense, BatchNormalization
from keras.api._v2.keras.optimizers import Adam

GAME_STATE_SIZE = 70
NUM_TOTAL_MOVES = 96


def main():

    # Parse flags
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_games",
        type=int,
        default=25000,
        help="Number of games used for training a generation",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1600,
        help="Number of iterations/searches per turn",
    )
    parser.add_argument(
        "--c_puct",
        type=float,
        default=1.0,
        help="c_puct for Monte Carlo Search Tree. Default is 1.0",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.25,
        help="Epsilon for Monte Carlo Search Tree. Default is 0.25",
    )
    parser.add_argument(
        "--num_test_games",
        type=int,
        default=400,
        help="Number of games used to test a generation. Default 400.",
    )
    parser.add_argument(
        "--test_threshold",
        type=float,
        default=0.5,
        help="Minimum score (exclusive) to become new best agent. Default is 0.5",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=1,
        help="Number of processes used for self play. Default 1.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.01,
        help="Learning rate for neural network. Default is 0.01",
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
        "--num_old_gens",
        type=int,
        default=20,
        help="Maximum number of training samples from older generations to use. Default is 20.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="",
        help="Name of the run. Will write outputs into folder {name}"
        "or continue a previous run if the folder exists."
        "Default is the empty string, in which case timestamp is used",
    )
    parser.add_argument(
        "--cwd",
        type=str,
        default=".",
        help="Folder in which all logging resides. Default is .",
    )
    parser.add_argument(
        "--num_logged",
        type=int,
        default=0,
        help="Number of games to log. Default is 0",
    )

    # Dictionary of flag values
    args = vars(parser.parse_args())

    PROCESSES = min(cpu_count(), max(1, args["processes"]))
    # Divisible by the number of processes
    NUM_GAMES = PROCESSES * (max(1, args["num_games"] // PROCESSES))
    ITERATIONS = max(2, args["iterations"])
    C_PUCT = max(0.0, args["c_puct"])
    EPSILON = min(1.0, max(0.0, args["epsilon"]))
    # Enforce even number (first player bias)
    NUM_TEST_GAMES = 2 * PROCESSES * max(1, args["num_test_games"] // (2 * PROCESSES))
    TEST_THRESHOLD = min(
        (NUM_TEST_GAMES - 0.5) / NUM_TEST_GAMES, max(0.5, args["test_threshold"])
    )
    LEARNING_RATE = max(0.0, args["learning_rate"])
    BATCH_SIZE = max(1, args["batch_size"])
    EPOCHS = max(1, args["epochs"])
    NUM_OLD_GENS = max(0, args["num_old_gens"])
    NAME = args["name"]
    NUM_LOGGED = max(0, args["num_logged"])

    # "Normalize" position where we create files, etc.
    cwd = args["cwd"]
    if not os.path.isdir(cwd):
        os.mkdir(cwd)

    # Find needed resources when the run already exists
    if len(NAME) > 0 and os.path.isdir(os.path.join(cwd, f"{NAME}")):
        current_generation = int(
            open(
                f"{cwd}/{NAME}/metadata/current_generation.txt",
                encoding="utf-8",
            )
            .read()
            .strip()
        )
        best_generation = int(
            open(f"{cwd}/{NAME}/metadata/best_generation.txt", encoding="utf-8")
            .read()
            .strip()
        )
        cur_gen_location = f"{cwd}/{NAME}/generations/gen_{current_generation}/model"
        best_gen_location = f"{cwd}/{NAME}/generations/gen_{best_generation}/model"
        old_training_samples = []
        for gen in range(
            max(1, current_generation - NUM_OLD_GENS + 1), current_generation + 1
        ):
            location = f"{cwd}/{NAME}/generations/gen_{gen}/training_samples"
            if os.path.isdir(location):
                old_training_samples.append(location)

        # Create folder for new generation
        new_gen_folder = f"{cwd}/{NAME}/generations/gen_{current_generation+1}"
        # Reset folder if it already exists
        if os.path.isdir(new_gen_folder):
            shutil.rmtree(new_gen_folder)
        os.mkdir(new_gen_folder)

    # Start a new run
    else:

        # Generate name from timestamp if no name given
        if len(NAME) == 0:
            now = datetime.datetime.now()
            NAME = (
                f"_run_{now.year}{now.month}{now.day}{now.hour}{now.minute}{now.second}"
            )

        # Create folder, implausible that name is repeated
        os.mkdir(f"{cwd}/{NAME}")

        # Create model
        input_layer = Input(shape=(GAME_STATE_SIZE,))
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
        eval_output = Dense(units=1, activation="tanh")(layer_3)
        prob_output = Dense(units=NUM_TOTAL_MOVES, activation="softmax")(layer_3)

        model = Model(inputs=input_layer, outputs=[eval_output, prob_output])
        model.compile(
            optimizer=Adam(learning_rate=0.01),
            loss=[
                keras.losses.MeanSquaredError(),
                keras.losses.CategoricalCrossentropy(),
            ],
            jit_compile=True,
        )

        # Gen 0 only holdes the random weight neural network
        os.mkdir(f"{cwd}/{NAME}/generations")
        os.mkdir(f"{cwd}/{NAME}/generations/gen_0")
        model.save(f"{cwd}/{NAME}/generations/gen_0/model")

        os.mkdir(f"{cwd}/{NAME}/metadata")
        open(
            f"{cwd}/{NAME}/metadata/current_generation.txt",
            "w+",
            encoding="utf-8",
        ).write("0")
        open(
            f"{cwd}/{NAME}/metadata/best_generation.txt",
            "w+",
            encoding="utf-8",
        ).write("0")

        # Create folder for new generation
        new_gen_folder = f"{cwd}/{NAME}/generations/gen_1"
        os.mkdir(new_gen_folder)

        cur_gen_location = f"{cwd}/{NAME}/generations/gen_0/model"
        best_gen_location = f"{cwd}/{NAME}/generations/gen_0/model"
        old_training_samples = []

    new_model_location = f"{new_gen_folder}/model"
    train_log_folder = f"{new_gen_folder}/training_logs"
    test_log_folder = f"{new_gen_folder}/testing_logs"
    train_sample_folder = f"{new_gen_folder}/training_samples"
    os.mkdir(train_log_folder)
    os.mkdir(test_log_folder)
    os.mkdir(train_sample_folder)

    open(f"{new_gen_folder}/metadata.txt", "w+", encoding="utf-8").write(
        f"Number of games: {NUM_GAMES}\n"
        f"Number of iterations per turn: {ITERATIONS}\n"
        f"c_puct: {C_PUCT}\n"
        f"epsilon: {EPSILON}\n"
        f"Number of test games: {NUM_TEST_GAMES}\n"
        f"Number of processes used in self play: {PROCESSES}\n"
        f"Learning rate: {LEARNING_RATE}\n"
        f"Batch size used in training: {BATCH_SIZE}\n"
        f"Epochs used in training: {EPOCHS}\n"
        f"Old samples used: {old_training_samples}"
    )

    res = train_generation(
        cur_gen_location=cur_gen_location,
        best_gen_location=best_gen_location,
        new_model_location=new_model_location,
        train_log_folder=train_log_folder,
        test_log_folder=test_log_folder,
        train_sample_folder=train_sample_folder,
        num_games=NUM_GAMES,
        iterations=ITERATIONS,
        num_test_games=NUM_TEST_GAMES,
        num_logged=NUM_LOGGED,
        testing_threshold=TEST_THRESHOLD,
        c_puct=C_PUCT,
        epsilon=EPSILON,
        processes=PROCESSES,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        old_training_samples=old_training_samples,
    )

    current_generation = int(
        open(
            f"{cwd}/{NAME}/metadata/current_generation.txt",
            encoding="utf-8",
        )
        .read()
        .strip()
    )

    # Generation passed, update best gen
    if res:
        open(
            f"{cwd}/{NAME}/metadata/best_generation.txt",
            "w+",
            encoding="utf-8",
        ).write(f"{current_generation+1}")
    open(
        f"{cwd}/{NAME}/metadata/current_generation.txt",
        "w+",
        encoding="utf-8",
    ).write(f"{current_generation+1}")


if __name__ == "__main__":
    main()
