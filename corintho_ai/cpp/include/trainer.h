#ifndef TRAINER_H
#define TRAINER_H

#include <memory>
#include <random>
#include <string>
#include <vector>

#include "util.h"

class SelfPlayer;

/// @brief Orchestrates many SelfPlayer objects to generate training samples
/// from self-play games
/// @details This is the class that is used by Cython
class Trainer {
 public:
  /// @brief Default constructor
  /// This is needed for Cython to be able to create a Trainer object
  Trainer() = default;
  Trainer(int32_t num_games, const std::string &logging_folder, int32_t seed,
          int32_t max_searches = 1600, int32_t searches_per_eval = 16,
          float c_puct = 1.0, float epsilon = 0.25, int32_t num_logged = 10,
          int32_t num_threads = 1, bool testing = false);
  ~Trainer() = default;

  /// @brief Return the number of requests for evaluations
  int32_t numRequests(int32_t to_play = -1) const noexcept;
  /// @brief Return the number of training samples
  int32_t numSamples() const noexcept;
  /// @brief Average score of first player
  float score() const noexcept;
  /// @brief Return the average mate length
  float getAvgMateLen() const noexcept;

  /// @brief Write the game states for which evaluations are requested
  /// @param game_states The array to write the game states to
  /// @return The number of requests for evaluations
  void writeRequests(float *game_states, int32_t to_play = -1) const noexcept;
  /// @brief Write the training samples
  void writeSamples(float *game_states, float *eval_samples,
                    float *prob_samples) const noexcept;
  /// @brief Write a summary of the game outcomes to a file
  /// @param file The name of the file to write to
  /// @return The percentage score of the first player. This is used to
  /// determine if a generation improved
  float writeScores(const std::string &file) const;

  /// @brief This is the main function that runs the self-play games. It is
  /// called by Cython in a loop.
  /// @return If all games are done
  bool doIteration(float eval[], float probs[], int32_t to_play = -1);

 private:
  // Initialize SelfPlayers (factored out of different version of constructor)
  void initialize(int32_t num_games, const std::string &logging_folder,
                  int32_t max_searches, int32_t searches_per_eval, float c_puct,
                  float epsilon, int32_t num_logged, bool testing);

  /// @brief The self-play games
  std::vector<std::unique_ptr<SelfPlayer>> games_{};
  /// @brief Tracks which games are done
  std::vector<bool> is_done_{};
  /// @brief Maximum number of searches per turn for the players
  /// @details This is used to compute offsets for starting the games
  int32_t max_searches_{1600};
  /// @brief The maximum number of searches per neural network evaluation
  /// @details This is used to compute the size of arrays for neural network
  /// input and output
  int32_t searches_per_eval_{16};
  /// @brief The number of threads to use
  /// @details If 0, then use the number of threads available (which OpenMP will
  /// automatically do)
  int32_t num_threads_{0};
  /// @brief The number of searches done so far
  /// TODO: Is this searches or evaluations?
  int32_t searches_done_{0};
  /// @brief Random number generator
  std::mt19937 generator_{};
};

#endif