import toml
import argparse
import os

# Argument structure: config.py [program to be run] -p [config file path]
if __name__ == "__main__":

    # Parse flags
    parser = argparse.ArgumentParser()

    parser = argparse.ArgumentParser(description='Generates a command line function based on a configuration file.')

    parser.add_argument(
        "program",
        help="Path to the program to be run.",
    )

    parser.add_argument(
        "-p",
        "--path",
        help="Path to the configuration file.",
    )

    # Dictionary of flag values
    args = vars(parser.parse_args())

    program = args["program"]
    path = args["path"]

    commands = []

    with open(path,'r') as f:
        config = toml.load(f)
        hyperparameters = config["hyperparameters"]
        hyperparameters_flags = ' '.join(["--"+x+"="+str(hyperparameters[x]) for x in hyperparameters])
        commands.append('python '+program+' '+hyperparameters_flags)

    for element in commands:
        os.system(element)