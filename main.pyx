# distutils: language = c++

from libcpp.string cimport string
from libcpp cimport bool
import numpy as np
cimport numpy as np
import time
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import keras.api._v2.keras as keras
from keras.api._v2.keras.models import load_model
from keras import backend as K
from keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau

cdef extern from "cpp/trainer.cpp":
    cdef cppclass Trainer:
        Trainer()
        Trainer(int num_games, int num_logged, int num_iterations,
                float c_puct, float epsilon, int threads, int searches_per_eval, string logging_folder,
                int random_seed)
        Trainer(int num_games, int num_logged, int num_iterations,
                float c_puct, float epsilon, int threads, int searches_per_eval, string logging_folder,
                int random_seed, bool)
        bool do_iteration(float *evaluations, float *probabilities)
        bool do_iteration(float *evaluations, float *probabilities, int to_play)
        int write_requests(float *game_states)
        int write_requests(float *game_states, int to_play)
        int count_samples()
        void write_samples(float *game_states, float *evaluation_samples, float *probability_samples)
        float get_score(string out_file)
        float get_avg_mate_len()

def format_time(t):
    """Format string
    t is time in seconds"""
    if t < 1:
        return f"{t}s"
    if t < 60:
        return f"{t:.2f}s"
    if t < 3600:
        return f"{int(t//60)}m{round(t)%60:02d}s"
    return f"{int(t//3600)}h{int((t % 3600)//60):02d}m{round(t)%60:02d}s"

def write_loss(loss_csv, loss_file):
    """Append best loss to loss file"""

    best_loss = 999
    first = True
    for line in open(loss_csv, encoding="utf-8"):
        if first:
            first = False
            continue
        lst = line.split('\t')
        best_loss = min(best_loss, float(lst[-1]))
    
    open(loss_file, "a+", encoding="utf-8").write(f"{best_loss}\n")

def train_generation(*,
    cur_gen_location,  # neural network to train on
    best_gen_location,  # opponent to test against
    new_model_location,  # location to save new model (is this a folder or a file?)
    train_log_folder,
    test_log_folder,
    train_sample_folder,  # String path to write training samples into
    loss_file,  # String path to write loss into
    num_games=25000,
    iterations=1600,
    num_test_games=400,
    num_logged=10,
    testing_threshold=0.55,  # Non-inclusive lower bound for new generation to pass
    c_puct=1.0,
    epsilon=0.25,
    threads=1,
    searches_per_eval=1,
    learning_rate=0.01,
    batch_size=2048,
    epochs=1,
    anneal_factor=0.5,
    patience=3,
    old_training_samples=[],  # list of folders containing training sample files from previous generations to use
    best_gen_rating=0,
    new_rating_file="",
):

    cdef unsigned int NUM_MOVES = 96
    cdef unsigned int GAME_STATE_SIZE = 70
    cdef unsigned int SYMMETRY_NUM = 8
    EVALS_PER_SEARCH = 26.5

    rng = np.random.default_rng(int(time.time()))

    # Training

    # Load playing model
    model = load_model(best_gen_location)

    cdef Trainer* trainer = new Trainer(
        num_games,
        num_logged,
        iterations,
        c_puct,
        epsilon,
        threads,
        searches_per_eval,
        train_log_folder.encode(),
        rng.integers(65536),  # Random seed
    )

    cdef np.ndarray[np.float32_t, ndim=1] evaluations = np.zeros(num_games * searches_per_eval, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] probabilities = np.zeros((num_games * searches_per_eval, NUM_MOVES), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] game_states = np.zeros((num_games * searches_per_eval, GAME_STATE_SIZE), dtype=np.float32)

    cdef int evaluations_done = 0

    print("Begin training!")

    start_time = time.perf_counter()
    last_time = start_time

    # First iteration
    trainer.do_iteration(&evaluations[0], &probabilities[0,0])

    predict_time = 0
    play_time = 0

    while True:

        num_requests = trainer.write_requests(&game_states[0,0])

        pred_start = time.perf_counter()
        # use_multiprocessing probably does nothing
        # but it does not hurt
        # and in one case it was removed and NN evals became much slower
        res = model.predict(
            x=game_states, batch_size=num_requests, verbose=0, steps=1,
        )
        evaluations = res[0].flatten()
        probabilities = res[1]
        predict_time += time.perf_counter()-pred_start

        play_start = time.perf_counter()
        res = trainer.do_iteration(&evaluations[0], &probabilities[0,0])
        play_time += time.perf_counter()-play_start

        if res:
            break

        evaluations_done += 1

        # Log every minute
        if time.perf_counter() - last_time > 60:
            time_taken = time.perf_counter() - start_time
            open(f"{train_log_folder}/progress.txt", 'a+', encoding='utf-8').write(
                f"{evaluations_done} evaluations completed in {format_time(time_taken)}\n"
                f"Predicted time to complete: {format_time(EVALS_PER_SEARCH*iterations/searches_per_eval*time_taken/evaluations_done)}\n"
                f"Estimated time left: {format_time((EVALS_PER_SEARCH*iterations/searches_per_eval-evaluations_done)*time_taken/evaluations_done)}\n"
                f"Prediction time so far: {format_time(predict_time)}\n"
                f"Play time so far: {format_time(play_time)}\n\n"
            )
            last_time = time.perf_counter()

    time_taken = time.perf_counter() - start_time
    open(f"{train_log_folder}/progress.txt", 'a+', encoding='utf-8').write(
        "Training Complete!\n"
        f"{evaluations_done} evaluations completed in {format_time(time_taken)}\n"
        f"Total prediction time: {format_time(predict_time)}\n"
        f"Total play time: {format_time(play_time)}\n"
    )

    # Get training samples
    num_samples = trainer.count_samples()
    cdef np.ndarray[np.float32_t, ndim=2] sample_states = np.zeros((num_samples * SYMMETRY_NUM, GAME_STATE_SIZE), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] evaluation_labels = np.zeros(num_samples * SYMMETRY_NUM, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] probability_labels = np.zeros((num_samples * SYMMETRY_NUM, NUM_MOVES), dtype=np.float32)
    trainer.write_samples(&sample_states[0,0], &evaluation_labels[0], &probability_labels[0,0])

    open(f"{train_log_folder}/self_play_time.txt", "w+", encoding='utf-8').write(
        f"{num_games} games played\n"
        f"{iterations} searches per turn\n"
        f"Training complete in {format_time(time_taken)}\n"
        f"{format_time(predict_time)} on neural network predictions\n"
        f"{evaluations_done} evaluations\n"
        f"{format_time(time_taken/evaluations_done)} per evaluation\n"
        f"{format_time(play_time)} on self play\n"
        f"{num_samples} total turns\n"
        f"{num_samples/num_games} average turns per game\n"
        f"{format_time(play_time/num_samples)} per turn\n"
        f"{format_time(play_time/(num_samples*iterations))} per search\n"
        f"{trainer.get_avg_mate_len()} average mate length\n"
    )

    trainer.get_score(f"{train_log_folder}/score_verbose.txt".encode())

    del trainer

    # Save training samples
    np.savez_compressed(f"{train_sample_folder}/game_states", sample_states)
    np.savez_compressed(f"{train_sample_folder}/evaluation_labels", evaluation_labels)
    np.savez_compressed(f"{train_sample_folder}/probability_labels", probability_labels)

    # Add old training samples
    for cur_path in old_training_samples:
        old_sample_states = np.load(f"{cur_path}/game_states.npz")
        old_evaluation_labels = np.load(f"{cur_path}/evaluation_labels.npz")
        old_probability_labels = np.load(f"{cur_path}/probability_labels.npz")
        np.concatenate((sample_states, np.reshape(old_sample_states["arr_0"], (-1, GAME_STATE_SIZE))))
        np.concatenate((evaluation_labels, old_evaluation_labels["arr_0"]))
        np.concatenate((probability_labels, np.reshape(old_probability_labels["arr_0"], (-1, NUM_MOVES))))
        old_sample_states.close()
        old_evaluation_labels.close()
        old_probability_labels.close()

    # Load training model
    training_model = load_model(cur_gen_location)
    # Set learning rate
    K.set_value(training_model.optimizer.learning_rate, learning_rate)
    # Train neural net
    checkpoint = ModelCheckpoint(
        new_model_location,
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,
        mode='auto',
        save_frequency=1,
    )
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=anneal_factor, patience=patience)
    csv_logger = CSVLogger(f"{train_log_folder}/train_loss.csv", separator='\t')

    fit_start_time = time.perf_counter()

    training_model.fit(
        x=sample_states,
        y=[evaluation_labels, probability_labels],
        batch_size=batch_size,
        epochs=epochs,
        shuffle=True,
        validation_split=0.3,
        callbacks=[checkpoint, reduce_lr, csv_logger],
        verbose=0,
    )

    time_taken = time.perf_counter() - fit_start_time
    open(f"{train_log_folder}/fit_time.txt", "w+", encoding='utf-8').write(
        f"Neural network fitting completed in {format_time(time_taken)}\n"
        f"{num_samples} samples\n"
        f"{batch_size} batch size\n"
        f"{epochs} epochs\n"
        f"{format_time(time_taken/epochs)} per epoch\n"
        f"{format_time(time_taken/(epochs*num_samples/batch_size))} per batch\n"
    )

    write_loss(f"{train_log_folder}/train_loss.csv", loss_file)

    print(f"Training took {format_time(time.perf_counter() - start_time)}!")

    # Testing

    new_model = load_model(new_model_location)

    cdef Trainer* tester = new Trainer(
        num_test_games,
        num_logged,
        iterations,
        c_puct,
        epsilon,
        threads,
        searches_per_eval,
        test_log_folder.encode(),
        rng.integers(65536),  # Random seed
        True,  # Testing
    )

    cdef np.ndarray[np.float32_t, ndim=1] evaluations_test = np.zeros(num_test_games * searches_per_eval, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] probabilities_test = np.zeros((num_test_games * searches_per_eval, NUM_MOVES), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] test_game_states = np.zeros((num_test_games * searches_per_eval, GAME_STATE_SIZE), dtype=np.float32)

    evaluations_done = 0

    print("Begin testing!")

    start_time = time.perf_counter()
    last_time = start_time

    predict_time = 0
    play_time = 0
    to_play = 0
    done = False

     # First iteration
    tester.do_iteration(&evaluations_test[0], &probabilities_test[0,0], to_play)

    while not done:

        num_requests = tester.write_requests(&test_game_states[0,0], to_play)

        if num_requests > 0:
            pred_start = time.perf_counter()
            if to_play == 0:
                res = new_model.predict(
                    x=test_game_states, batch_size=num_requests, verbose=0, steps=1,
                )
            else:
                res = model.predict(
                    x=test_game_states, batch_size=num_requests, verbose=0, steps=1,
                )
            evaluations_test = res[0].flatten()
            probabilities_test = res[1]
            predict_time += time.perf_counter()-pred_start

            play_start = time.perf_counter()
            res = tester.do_iteration(&evaluations_test[0], &probabilities_test[0,0],
                to_play,
            )
            play_time += time.perf_counter() - play_start
            if res:
                done = True
                break

            evaluations_done += 1
        else:
            to_play = 1 - to_play

        if time.perf_counter() - last_time > 60:
            time_taken = time.perf_counter() - start_time
            open(f"{test_log_folder}/progress.txt", 'a+', encoding='utf-8').write(
                f"{evaluations_done} evaluations completed in {format_time(time_taken)}\n"
                f"Predicted time to complete: {format_time(EVALS_PER_SEARCH*iterations/searches_per_eval*time_taken/evaluations_done)}\n"
                f"Estimated time left: {format_time((EVALS_PER_SEARCH*iterations/searches_per_eval-evaluations_done)*time_taken/evaluations_done)}\n"
                f"Prediction time so far: {format_time(predict_time)}\n"
                f"Play time so far: {format_time(play_time)}\n\n"
            )
            last_time = time.perf_counter()

    time_taken = time.perf_counter() - start_time

    open(f"{test_log_folder}/test_time.txt", "w+", encoding='utf-8').write(
        f"Testing complete in {format_time(time_taken)}\n"
        f"{num_test_games} games\n"
        f"Total prediction time: {format_time(predict_time)}\n"
        f"Total play time: {format_time(play_time)}\n"
    )

    # Clear old models
    keras.backend.clear_session()

    score = tester.get_score(f"{test_log_folder}/score_verbose.txt".encode())

    del tester

    open(f"{test_log_folder}/progress.txt", 'a+', encoding='utf-8').write(
        "Testing Complete!\n"
        f"{evaluations_done} evaluations completed in {format_time(time_taken)}\n"
        f"Total prediction time: {format_time(predict_time)}\n"
        f"Total play time: {format_time(play_time)}\n"
        f"New agent score {score:1f}!\n"
    )

    open(f"{test_log_folder}/score.txt", 'w', encoding='utf-8').write(f"New agent score {score:1f}!\n")

    # Update rating
    if score > 0:
        new_rating = best_gen_rating - 400 * np.log10(1 / score - 1)
    else:
        new_rating = best_gen_rating - 400
    open(new_rating_file, 'w+', encoding='utf-8').write(f"{new_rating}\n")


    # Track success and update ratin
    if score > testing_threshold:
        return True

    return False


