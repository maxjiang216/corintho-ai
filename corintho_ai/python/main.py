import toml
import argparse
import os

# Argument structure: config.py [program to be run] -p [config file path]
if __name__ == "__main__":
    # Parse flags
    parser = argparse.ArgumentParser()

    parser = argparse.ArgumentParser(
        description="Generates a command line function "
        "based on a configuration file."
    )

    parser.add_argument(
        "program",
        type=str,
        help="Path to the program to be run.",
    )

    parser.add_argument(
        "-p",
        "--path",
        type=str,
        help="Path to the configuration file.",
    )

    parser.add_argument(
        "-n",
        "--num_generations",
        type=int,
        default=1,
        help="Number of generations to train",
    )

    # Dictionary of flag values
    args = vars(parser.parse_args())

    program = args["program"]
    path = args["path"]
    num_gens = args["num_generations"]

    commands = []

    with open(path, "r") as f:
        config = toml.load(f)
        hyperparameters = config["hyperparameters"]
        hyperparameters_flags = " ".join(
            ["--" + x + "=" + str(hyperparameters[x]) for x in hyperparameters]
        )
        for _ in range(num_gens):
            commands.append("python3 " + program + " " + hyperparameters_flags)

    for element in commands:
        os.system(element)
