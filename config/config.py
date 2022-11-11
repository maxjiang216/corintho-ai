import toml
import sys
import os

# Argument structure: config.py [program to be run] [config file path]

if (len(sys.argv) < 3):
    raise Exception("The name of the program and path to the configuration file are required.\n")

program = sys.argv[1]
path = sys.argv[2]

with open(path,'r') as f:
    config = toml.load(f)
    hyperparameters = config["hyperparameters"]
    hyperparameters_flags = ' '.join(["--"+x+"="+str(hyperparameters[x]) for x in hyperparameters])
    os.system("python " + program + ' ' + hyperparameters_flags)