import argparse
import json
import os
import shutil
from datetime import datetime
from multiprocessing import cpu_count

import keras.api._v2.keras as keras
import pytz
from corintho import train_generation
from keras.api._v2.keras import Input, regularizers
from keras.api._v2.keras.layers import Activation, BatchNormalization, Dense
from keras.api._v2.keras.models import Model
from keras.api._v2.keras.optimizers import Adam

_GAME_STATE_SIZE = 70
_NUM_MOVES = 96


def get_args():
    """Parse command line arguments"""
    # Parse flags
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--anneal_factor",
        type=float,
        default=0.5,
        help="Factor to reduce learning rate upon plateau. Default 0.5.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2048,
        help="Batch size for neural network training. Default 2048.",
    )
    parser.add_argument(
        "--c_puct",
        type=float,
        default=1.0,
        help="c_puct for Monte Carlo Search Tree. Default is 1.0",
    )
    parser.add_argument(
        "--cwd",
        type=str,
        default=".",
        help="Folder in which all logging resides. Default is .",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of epochs for neural network training. Default 1.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.25,
        help="Epsilon for Monte Carlo Search Tree. Default is 0.25",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.01,
        help="Learning rate for neural network. Default is 0.01",
    )
    parser.add_argument(
        "--max_searches",
        type=int,
        default=1600,
        help="Number of iterations/searches per turn",
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
        "--num_games",
        type=int,
        default=25000,
        help="Number of games used for training a generation",
    )
    parser.add_argument(
        "--num_logged",
        type=int,
        default=0,
        help="Number of games to log. Default is 0",
    )
    parser.add_argument(
        "--num_old_gens",
        type=int,
        default=20,
        help="Maximum number of training samples "
        "from older generations to use. Default is 20.",
    )
    parser.add_argument(
        "--num_test_games",
        type=int,
        default=400,
        help="Number of games used to test a generation. Default 400.",
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=0,
        help="Number of threads used for self play. "
        "Default 0 in which case 2x the number of CPUs will be used.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Number of epochs before annealling learning rate. Default 3.",
    )
    parser.add_argument(
        "--searches_per_eval",
        type=int,
        default=1,
        help="Maximum number of searches needing evaluations to do "
        "before running a neural network evaluation. "
        "Default is 1, which is the standard MCST algorithm.",
    )
    parser.add_argument(
        "--test_threshold",
        type=float,
        default=0.5,
        help="Minimum score (exclusive) to become new best agent. "
        "Default is 0.5",
    )

    # Dictionary of flag values
    args = vars(parser.parse_args())

    # Process and validate the values
    args["anneal_factor"] = max(0.0, min(1.0, args["anneal_factor"]))
    args["batch_size"] = max(1, args["batch_size"])
    args["c_puct"] = max(0.0, args["c_puct"])
    args["epochs"] = max(1, args["epochs"])
    args["epsilon"] = min(1.0, max(0.0, args["epsilon"]))
    args["learning_rate"] = max(0.0, args["learning_rate"])
    args["max_searches"] = max(2, args["max_searches"])
    args["num_games"] = max(1, args["num_games"])
    args["num_logged"] = max(0, args["num_logged"])
    args["num_old_gens"] = max(0, args["num_old_gens"])
    # Enforce even number (first player bias)
    args["num_test_games"] = 2 * max(1, args["num_test_games"] // 2)
    args["num_threads"] = (
        args["num_threads"] if args["num_threads"] > 0 else 2 * cpu_count()
    )
    args["patience"] = max(1, min(args["epochs"], args["patience"]))
    args["searches_per_eval"] = min(
        args["max_searches"] - 1, max(1, args["searches_per_eval"])
    )
    args["test_threshold"] = min(
        (args["num_test_games"] - 0.5) / args["num_test_games"],
        max(0.5, args["test_threshold"]),
    )
    return args


def setup_existing_run(params):
    """
    Get metadata from existing run
    """
    cwd = params["cwd"]
    name = params["name"]
    num_old_gens = params["num_old_gens"]

    current_generation = int(
        open(
            f"{cwd}/{name}/metadata/current_generation.txt",
            encoding="utf-8",
        )
        .read()
        .strip()
    )
    best_generation = int(
        open(f"{cwd}/{name}/metadata/best_generation.txt", encoding="utf-8")
        .read()
        .strip()
    )
    params[
        "cur_gen_location"
    ] = f"{cwd}/{name}/generations/gen_{current_generation}/model"
    params[
        "best_gen_location"
    ] = f"{cwd}/{name}/generations/gen_{best_generation}/model"
    old_training_samples = []
    for gen in range(
        max(1, current_generation - num_old_gens + 1),
        current_generation + 1,
    ):
        location = f"{cwd}/{name}/samples/gen_{gen}"
        if os.path.isdir(location):
            old_training_samples.append(location)
    params["old_training_samples"] = old_training_samples
    # Get learning rate
    params["learning_rate"] = float(
        open(
            f"{cwd}/{name}/metadata/learning_rate.txt",
            encoding="utf-8",
        )
        .read()
        .strip()
    )
    params["best_gen_rating"] = float(
        open(
            f"{cwd}/{name}/generations/gen_{best_generation}/rating.txt",
            encoding="utf-8",
        )
        .read()
        .strip()
    )

    # Create folder for new generation
    new_gen_folder = f"{cwd}/{name}/generations/gen_{current_generation+1}"
    # Reset folder if it already exists
    if os.path.isdir(new_gen_folder):
        shutil.rmtree(new_gen_folder)
    os.mkdir(new_gen_folder)
    new_gen_samples = f"{cwd}/{name}/samples/gen_{current_generation+1}"
    # Reset folder if it already exists
    if os.path.isdir(new_gen_samples):
        shutil.rmtree(new_gen_samples)
    os.mkdir(new_gen_samples)

    params[
        "new_rating_file"
    ] = f"{cwd}/{name}/generations/gen_{current_generation+1}/rating.txt"
    params["current_generation"] = current_generation
    params["best_generation"] = best_generation
    params["new_gen_samples"] = new_gen_samples

    return params


def setup_new_run(params):
    """ """
    name = params["name"]
    cwd = params["cwd"]
    learning_rate = params["learning_rate"]
    # Generate name from timestamp if no name given
    # implausible that name is repeated
    if len(name) == 0:
        now = datetime.now()
        name = f"_run_{now.year}{now.month}{now.day}"
        f"{now.hour}{now.minute}{now.second}"

    # Create folder
    os.mkdir(f"{cwd}/{name}")

    # Create model
    input_layer = Input(shape=(_GAME_STATE_SIZE,))
    prev_layer = input_layer
    for _ in range(12):
        new_layer = BatchNormalization()(
            Activation("relu")(
                Dense(
                    units=100,
                    kernel_regularizer=regularizers.L1L2(),
                )(prev_layer)
            )
        )
        prev_layer = new_layer
    eval_output = Dense(units=1, activation="tanh")(prev_layer)
    prob_output = Dense(units=_NUM_MOVES, activation="softmax")(prev_layer)

    model = Model(inputs=input_layer, outputs=[eval_output, prob_output])
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=[
            keras.losses.MeanSquaredError(),
            keras.losses.CategoricalCrossentropy(),
        ],
        loss_weights=[
            1.0,
            0.25,  # This is approximately |1 / (log(1 / 56))|
        ],
    )

    # Gen 0 only holdes the random weight neural network
    os.mkdir(f"{cwd}/{name}/generations")
    os.mkdir(f"{cwd}/{name}/generations/gen_0")
    model.save(f"{cwd}/{name}/generations/gen_0/model")

    os.mkdir(f"{cwd}/{name}/samples")

    # Initialize rating to 100
    params["best_gen_rating"] = 100
    open(
        f"{cwd}/{name}/generations/gen_0/rating.txt",
        "w+",
        encoding="utf-8",
    ).write("100")

    params["new_rating_file"] = f"{cwd}/{name}/generations/gen_1/rating.txt"

    # Initialize metadata
    os.mkdir(f"{cwd}/{name}/metadata")
    open(
        f"{cwd}/{name}/metadata/current_generation.txt",
        "w+",
        encoding="utf-8",
    ).write("0")
    open(
        f"{cwd}/{name}/metadata/best_generation.txt",
        "w+",
        encoding="utf-8",
    ).write("0")
    open(
        f"{cwd}/{name}/metadata/learning_rate.txt",
        "w+",
        encoding="utf-8",
    ).write(f"{learning_rate}")
    open(
        f"{cwd}/{name}/metadata/fails.txt",
        "w+",
        encoding="utf-8",
    ).write("0")

    # Create folder for new generation
    new_gen_folder = f"{cwd}/{name}/generations/gen_1"
    os.mkdir(new_gen_folder)

    new_gen_samples = f"{cwd}/{name}/samples/gen_1"
    os.mkdir(new_gen_samples)

    params["cur_gen_location"] = f"{cwd}/{name}/generations/gen_0/model"
    params["best_gen_location"] = f"{cwd}/{name}/generations/gen_0/model"
    params["old_training_samples"] = []
    params["current_generation"] = 0
    params["best_generation"] = 0
    params["new_gen_samples"] = new_gen_samples

    return params


def start_generation(params):
    """
    Set up and start a new generation
    """
    cwd = params["cwd"]
    name = params["name"]
    current_generation = params["current_generation"]
    new_gen_samples = params["new_gen_samples"]

    new_gen_folder = f"{cwd}/{name}/generations/gen_{current_generation+1}"

    params["new_model_location"] = f"{new_gen_folder}/model"
    train_log_folder = f"{new_gen_folder}/training_logs"
    test_log_folder = f"{new_gen_folder}/testing_logs"
    params["sample_folder"] = new_gen_samples
    os.mkdir(train_log_folder)
    os.mkdir(test_log_folder)

    # Add the current time to the dictionary
    params["start_time"] = str(
        datetime.now(tz=pytz.timezone("America/Toronto"))
    )
    with open(
        f"{new_gen_folder}/metadata.txt", "w+", encoding="utf-8"
    ) as file:
        json.dump(params, file, ensure_ascii=False, indent=4)

    params["loss_file"] = f"{cwd}/{name}/metadata/losses.txt"
    params["train_log_folder"] = train_log_folder
    params["test_log_folder"] = test_log_folder

    return train_generation(params)


def update_run_data(params, res):
    """
    Post-generation processing
    """
    cwd = params["cwd"]
    name = params["name"]
    current_generation = int(
        open(
            f"{cwd}/{name}/metadata/current_generation.txt",
            encoding="utf-8",
        )
        .read()
        .strip()
    )

    # Delete previous zip logs if it exists
    if os.path.isdir(f"{cwd}/zips"):
        shutil.rmtree(f"{cwd}/zips")
    # Recreate folder
    os.mkdir(f"{cwd}/zips")
    # Zip logs
    shutil.make_archive(
        f"{cwd}/zips/{name}_{current_generation+1}",
        "zip",
        f"{cwd}/{name}",
    )

    best_generation = params["best_generation"]
    # Generation passed, update best gen
    if res:
        best_generation = current_generation + 1
        open(
            f"{cwd}/{name}/metadata/best_generation.txt",
            "w+",
            encoding="utf-8",
        ).write(f"{best_generation}")

    current_generation += 1
    open(
        f"{cwd}/{name}/metadata/current_generation.txt",
        "w+",
        encoding="utf-8",
    ).write(f"{current_generation}")

    # We may want to update the learning rate
    write_learning_rate(
        best_generation,
        current_generation,
        f"{cwd}/{name}/metadata/losses.txt",
        f"{cwd}/{name}/metadata/learning_rate.txt",
        f"{cwd}/{name}/metadata/fails.txt",
        params["learning_rate"],
        params["patience"],
        params["anneal_factor"],
    )


def write_learning_rate(
    best_generation,
    current_generation,
    loss_file,
    learning_rate_file,
    fail_num_file,
    learning_rate,
    patience,
    factor,
):
    """
    Updates learning rate and writes into file
    """

    # If the generation passed
    if best_generation == current_generation:
        open(fail_num_file, "w+", encoding="utf-8").write("0")
        return

    # Otherwise, check for loss decrease
    min_loss = 999
    loss_fails = 0
    for line in open(loss_file, encoding="utf-8"):
        cur_loss = float(line.strip())
        if cur_loss < min_loss:
            min_loss = cur_loss
            loss_fails = 0
        else:
            loss_fails += 1

    # Loss improved
    if loss_fails == 0:
        open(fail_num_file, "w+", encoding="utf-8").write("0")
        return

    # This generation failed
    fail_num = (
        int(
            open(
                fail_num_file,
                encoding="utf-8",
            )
            .read()
            .strip()
        )
        + 1
    )

    # If too many failures, update learning rate
    if fail_num >= patience:
        open(learning_rate_file, "w+", encoding="utf-8").write(
            f"{learning_rate * factor}"
        )
        open(fail_num_file, "w+", encoding="utf-8").write("0")
    else:
        open(fail_num_file, "w+", encoding="utf-8").write(f"{fail_num}")


def main():
    params = get_args()

    # "Normalize" position where we create files, etc.
    cwd = params["cwd"]
    if not os.path.isdir(cwd):
        os.mkdir(cwd)

    name = params["name"]
    # Find needed resources when the run already exists
    if len(name) > 0 and os.path.isdir(os.path.join(cwd, f"{name}")):
        params = setup_existing_run(params)
    else:  # Start a new run
        params = setup_new_run(params)

    res = start_generation(params)

    update_run_data(params, res)


if __name__ == "__main__":
    main()
