# distutils: language = c++

import os
import time

import keras.api._v2.keras as keras
import numpy as np
from keras import backend as K
from keras.api._v2.keras.models import load_model
from keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau

cimport numpy as np
from libcpp cimport bool
from libcpp.string cimport string


cdef extern from "../cpp/src/trainer.cpp":
    cdef cppclass Trainer:
        Trainer(
            int num_games,
            string log_folder,
            int seed,
            int max_searches,
            int searches_per_eval,
            float c_puct,
            float epsilon,
            int num_logged,
            int num_threads,
            bool testing,
        ) except +
        int num_requests(int to_play) except +
        int num_samples() except +
        float score() except +
        float avg_mate_length() except +
        void writeRequests(float *game_states, int to_play) except +
        void writeSamples(float *game_states, float *eval_samples, float *prob_samples) except +
        void writeScores(string file) except +
        bool doIteration(float *evaluations, float *probabilities, int to_play) except +

cdef int _NUM_MOVES = 96
cdef int _GAME_STATE_SIZE = 70
cdef int _SYMMETRY_NUM = 8
cdef float _AVG_EVALS_PER_SEARCH = 26.5

cpdef format_time(t):
    """Format string
    t is time in seconds"""
    if t < 1:
        return f"{t}s"
    if t < 60:
        return f"{t:.2f}s"
    if t < 3600:
        return f"{int(t//60)}m{round(t)%60:02d}s"
    return f"{int(t//3600)}h{int((t % 3600)//60):02d}m{round(t)%60:02d}s"

cpdef write_loss(loss_csv, loss_file):
    """Append best loss to loss file"""
    best_loss = 999
    first = True
    with open(loss_csv, encoding="utf-8") as f:
        for line in f:
            if first:
                first = False
                continue
            lst = line.split('\t')
            best_loss = min(best_loss, float(lst[-1]))
    with open(loss_file, "a+", encoding="utf-8") as f:
        f.write(f"{best_loss}\n")

cpdef get_predictions(best_model, new_model, game_states, evals, probs, num_requests, to_play):
    """
    Get predictions from model
    """
    if new_model is not None and to_play == 0:
        res = new_model.predict(
            x=game_states, batch_size=num_requests, verbose=0, steps=1,
        )
    else:
        res = best_model.predict(
            x=game_states, batch_size=num_requests, verbose=0, steps=1,
        )
    evals[0:num_requests] = res[0].flatten()
    probs[:num_requests, :] = res[1]

cdef void log_stats(Trainer *trainer, time_taken, predict_time, play_time, evals_done, log_folder, params, testing=False):
    """
    Log some stats
    """
    num_games = params["num_games"]
    max_searches = params["max_searches"]

    with open(f"{log_folder}/progress.txt", 'a+', encoding='utf-8') as f:
        f.write(
            f"Complete!\n"
            f"{evals_done} evaluations completed in {format_time(time_taken)}\n"
            f"{format_time(time_taken / evals_done)} per evaluation\n"
            f"Total prediction time: {format_time(predict_time)}\n"
            f"Total play time: {format_time(play_time)}\n"
        )

    with open(f"{log_folder}/play_time.txt", "w+", encoding='utf-8') as f:
        f.write(
            f"{num_games} games played\n"
            f"{max_searches} searches per turn\n"
            f"Training complete in {format_time(time_taken)}\n"
            f"{format_time(predict_time)} for neural network predictions\n"
            f"{evals_done} evaluations\n"
            f"{format_time(time_taken / evals_done)} per evaluation\n"
            f"{format_time(play_time)} for self play\n"   
        )
        if not testing:
            num_samples = trainer.num_samples()
            f.write(
                f"{trainer.avg_mate_length()} average mate length\n"
                f"{num_samples} total turns\n"
                f"{num_samples / num_games} average turns per game\n"
                f"{format_time(play_time / num_samples)} per turn\n"
                f"{format_time(play_time / (num_samples * max_searches))} per search\n"
            )

    trainer.writeScores(f"{log_folder}/score_verbose.txt".encode())

cdef void play_games(Trainer *trainer, log_folder, params, best_model, new_model=None):
    """
    Main playing loop
    This is used in both training and testing
    """
    num_games = params["num_games"]
    searches_per_eval = params["searches_per_eval"]
    max_searches = params["max_searches"]

    cdef np.ndarray[np.float32_t, ndim=1] evals = np.zeros(num_games * searches_per_eval, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] probs = np.zeros((num_games * searches_per_eval, _NUM_MOVES), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] game_states = np.zeros((num_games * searches_per_eval, _GAME_STATE_SIZE), dtype=np.float32)
    to_play = -1 if new_model is None else 0
    predict_time = 0.0
    play_time = 0.0
    evals_done = 0
    start_time = time.perf_counter()
    last_time = start_time

    while True:
        play_start = time.perf_counter()
        res = trainer.doIteration(&evals[0], &probs[0,0], to_play)
        play_time += time.perf_counter() - play_start

        if res:
            break

        num_requests = trainer.num_requests(to_play)
        if num_requests == 0:
            if to_play != -1:
                to_play = 1 - to_play
                continue
            # Since during training we keep searching until there is at least one request,
            # we should never get here.
            # It is possible that a whole turn will be played without needing to evaluate,
            # but in that case we will continue to the next turn.
            # This occurs until the game is complete.
            # If all games are complete, the loop should have ended already.
            else:
                print(f"evals done: {evals_done}")
                raise Exception("No requests during training")
        
        pred_start = time.perf_counter()
        get_predictions(best_model, new_model, game_states, evals, probs, num_requests, to_play)
        predict_time += time.perf_counter() - pred_start

        evals_done += 1

        if time.perf_counter() - last_time > 60:
            time_taken = time.perf_counter() - start_time
            with open(f"{log_folder}/progress.txt", 'a+', encoding='utf-8') as f:
                f.write(
                    f"{evals_done} evaluations completed in {format_time(time_taken)}\n"
                    f"Predicted time to complete: "
                    f"{format_time(_AVG_EVALS_PER_SEARCH * max_searches / searches_per_eval * time_taken / evals_done)}\n"
                    f"Estimated time left: "
                    f"{format_time((_AVG_EVALS_PER_SEARCH * max_searches / searches_per_eval - evals_done) * time_taken /evals_done)}\n"
                    f"Neural network prediction time so far: {format_time(predict_time)}\n"
                    f"Self play time so far: {format_time(play_time)}\n\n"
                )
            last_time = time.perf_counter()

    time_taken = time.perf_counter() - start_time
    log_stats(trainer, time_taken, predict_time, play_time, evals_done, log_folder, params, new_model is not None)

cdef get_samples(Trainer *trainer, params):
    """
    Get training samples
    """
    # Get training samples
    num_samples = trainer.num_samples()
    cdef np.ndarray[np.float32_t, ndim=2] game_states = np.zeros((num_samples * _SYMMETRY_NUM, _GAME_STATE_SIZE), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] eval_labels = np.zeros(num_samples * _SYMMETRY_NUM, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] prob_labels = np.zeros((num_samples * _SYMMETRY_NUM, _NUM_MOVES), dtype=np.float32)
    trainer.writeSamples(&game_states[0,0], &eval_labels[0], &prob_labels[0,0])

    # Save training samples
    sample_folder = params["sample_folder"]
    np.savez_compressed(f"{sample_folder}/game_states", game_states)
    np.savez_compressed(f"{sample_folder}/evaluation_labels", eval_labels)
    np.savez_compressed(f"{sample_folder}/probability_labels", prob_labels)

    # Add old training samples
    old_training_samples = params["old_training_samples"]
    for cur_path in old_training_samples:
        old_game_states = np.load(f"{cur_path}/game_states.npz")
        old_eval_labels = np.load(f"{cur_path}/evaluation_labels.npz")
        old_prob_labels = np.load(f"{cur_path}/probability_labels.npz")
        np.concatenate((game_states, np.reshape(old_game_states["arr_0"], (-1, _GAME_STATE_SIZE))))
        np.concatenate((eval_labels, old_eval_labels["arr_0"]))
        np.concatenate((prob_labels, np.reshape(old_prob_labels["arr_0"], (-1, _NUM_MOVES))))
        old_game_states.close()
        old_eval_labels.close()
        old_prob_labels.close()

    return game_states, eval_labels, prob_labels

cdef void train_neural_network(Trainer *trainer, game_states, eval_labels, prob_labels, params):
    """
    Train neural network
    """
    # Load training model
    model = load_model(params["cur_gen_location"])
    # Set learning rate
    K.set_value(model.optimizer.learning_rate, params["learning_rate"])
    # Train neural net
    checkpoint = ModelCheckpoint(
        params["new_model_location"],
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=False,
        mode="auto",
        save_frequency=1,
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=params["anneal_factor"],
        patience=params["patience"],
    )
    log_folder = params["train_log_folder"]
    csv_logger = CSVLogger(f"{log_folder}/train_loss.csv", separator='\t')

    start_time = time.perf_counter()
    batch_size = params["batch_size"]
    epochs = params["epochs"]
    model.fit(
        x=game_states,
        y=[eval_labels, prob_labels],
        batch_size=batch_size,
        epochs=epochs,
        shuffle=True,
        validation_split=0.3,
        callbacks=[checkpoint, reduce_lr, csv_logger],
        verbose=0,
    )
    time_taken = time.perf_counter() - start_time

    with open(f"{log_folder}/fit_time.txt", "w+", encoding='utf-8') as f:
        num_samples = trainer.num_samples()
        f.write(
            f"Neural network fitting completed in {format_time(time_taken)}\n"
            f"{num_samples} samples\n"
            f"{batch_size} batch size\n"
            f"{epochs} epochs\n"
            f"{format_time(time_taken / epochs)} per epoch\n"
            f"{format_time(time_taken / (epochs * num_samples / batch_size))} per batch\n"
        )

    write_loss(f"{log_folder}/train_loss.csv", params["loss_file"])

cpdef update_rating(new_rating_file, best_gen_rating, score):
    """
    Calculate new rating
    """
    if score > 0:
        new_rating = best_gen_rating - 400 * np.log10(1 / score - 1)
    else:
        new_rating = best_gen_rating - 400
    with open(new_rating_file, 'w+', encoding='utf-8') as f:
        f.write(f"{new_rating}\n")
        
def train_generation(params):
    """
    Train a generation

    Returns whether the new model improved
    """

    # Initialize random number generator
    rng = np.random.default_rng(int(time.time()))

    # Load playing model
    best_model = load_model(params["best_gen_location"])

    # Self play sample generation
    cdef Trainer* trainer = new Trainer(
        params["num_games"],
        params["train_log_folder"].encode(),
        rng.integers(65536),  # Random seed
        params["max_searches"],
        params["searches_per_eval"],
        params["c_puct"],
        params["epsilon"],
        params["num_logged"],
        params["num_threads"],
        False,  # Training
    )
    # Self play
    play_games(
        trainer,
        params["train_log_folder"],
        params,
        best_model,
    )
    print(f"Self play complete!")
    # Get training samples
    game_states, evaluation_labels, probability_labels = get_samples(trainer, params)

    # Training
    train_neural_network(trainer, game_states, evaluation_labels, probability_labels, params)

    del trainer

    # Testing
    new_model = load_model(params["new_model_location"])

    cdef Trainer* tester = new Trainer(
        params["num_test_games"],
        params["test_log_folder"].encode(),
        rng.integers(65536),  # Random seed
        params["max_searches"],
        params["searches_per_eval"],
        params["c_puct"],
        params["epsilon"],
        params["num_logged"],
        params["num_threads"],
        True,  # Testing
    )

    test_log_folder = params["test_log_folder"]
    play_games(
        tester,
        test_log_folder,
        params,
        best_model,
        new_model,
    )
    print(f"Testing complete!")
    score = tester.score()
    open(f"{test_log_folder}/score.txt", 'w', encoding='utf-8').write(f"New agent score {score:1f}!\n")
    update_rating(
        params["new_rating_file"],
        params["best_gen_rating"],
        score,
    )
    del tester

    # Clear models
    keras.backend.clear_session()

    # Return whether new model improved
    if score > params["test_threshold"]:
        return True
    return False
