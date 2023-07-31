import argparse
import os

import toml


def main():
    # Parse flags
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-p",
        "--path",
        type=str,
        help="Path to the configuration file.",
    )
    parser.add_argument(
        "-n",
        "--num_rounds",
        type=int,
        default=1,
        help="Number of rounds to play",
    )

    args = vars(parser.parse_args())

    path = args["path"]
    num_rounds = args["num_rounds"]

    commands = []

    with open(path, "r") as f:
        config = toml.load(f)
        hyperparameters = config["hyperparameters"]
        hyperparameters_flags = " ".join(
            ["--" + x + "=" + str(hyperparameters[x]) for x in hyperparameters]
        )
        for _ in range(num_rounds):
            commands.append("python3 round.py " + hyperparameters_flags)

    for element in commands:
        os.system(element)


if __name__ == "__main__":
    main()
