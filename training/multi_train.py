import os
import shutil
import sys
import time
import numpy as np
from multiprocessing import Pool

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import keras.api._v2.keras as keras
from keras.api._v2.keras import Input, regularizers
from keras.api._v2.keras.models import Model
from keras.api._v2.keras.layers import Activation, Dense, BatchNormalization
from keras.api._v2.keras.optimizers import Adam
from trainer import Trainer
from tester import Tester

NUM_GAMES = 30
ITERATIONS = 20
NUM_TEST_GAMES = 40
BATCH_SIZE = 2048
EPOCHS = 1
PROCESSES = 2


def helper(player):
    """Helper function for training and testing"""
    print("START PLAYING")
    return 0


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

        # Training

        trainers = []
        logging = True

        print(1)

        for _ in range(PROCESSES):
            trainers.append(
                Trainer(
                    model=playing_model,
                    model_num=model_num,
                    num_games=NUM_GAMES // PROCESSES,
                    iterations=ITERATIONS,
                    logging=logging,
                )
            )
            logging = False

        print(trainers)

        pool = Pool(processes=PROCESSES)
        print(pool)
        res = pool.map(helper, range(PROCESSES))
        pool.close()

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

        training_model.save(f"./training/models/model_{model_num}/new_model")

        start_time = time.time()

        print(f"Begin testing generation {model_num}!")

        # Prepare log files
        open(
            f"./training/models/model_{model_num}/logs/trained_weights.txt",
            "w",
            encoding="utf-8",
        ).write(str(training_model.get_weights()))

        # Testing

        testers = []
        logging = True

        for _ in range(PROCESSES):
            testers.append(
                Tester(
                    model=training_model,
                    old_model=playing_model,
                    model_num=model_num,
                    num_games=NUM_TEST_GAMES // PROCESSES,
                    iterations=ITERATIONS,
                    logging=logging,
                )
            )
            logging = False

        pool = Pool(processes=PROCESSES)
        res = pool.map(helper, testers)
        pool.close()

        score = sum(res) / (PROCESSES * (NUM_TEST_GAMES // PROCESSES))

        open(
            f"./training/models/model_{model_num}/logs/testing_stats.txt",
            "a",
            encoding="utf-8",
        ).write(f"NEW MODEL SCORE: {score}\nTIMES FAILED: {fail_num}")

        # New neural net scores >50%
        if score > 0.5:
            playing_model = training_model
            fail_num = 0
        else:
            fail_num += 1

        model_num += 1
