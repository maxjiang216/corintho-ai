#include <cmath>
#include <cstdio>

#include <fstream>
#include <iostream>
#include <vector>

#include <omp.h>

using namespace std;

double compute_linear(const vector<double> &ratings, int p1, int p2,
                      double b) {
  return ratings[p1] - ratings[p2] + b;
}

double get_expected_score(const vector<double> &ratings, int p1, int p2,
                          double b) {
  double linear = compute_linear(ratings, p1, p2, b);
  return 1 / (1 + exp(-linear));
}

double compute_contribution(const vector<double> &ratings,
                            const vector<int> &matchup, int p1, int p2,
                            double b) {
  double linear = compute_linear(ratings, p1, p2, b);
  double s = matchup[0] + (double)matchup[1] / 2;
  int n = matchup[0] + matchup[1] + matchup[2];
  return s * linear - (double)n * log(1 + exp(linear));
}

double compute_log_likelihood(const vector<double> &ratings,
                              const vector<vector<vector<int>>> &matchups,
                              double b) {
  double log_likelihood = 0;
  for (size_t i = 0; i < matchups.size(); ++i) {
    for (size_t j = 0; j < matchups[i].size(); ++j) {
      log_likelihood += compute_contribution(ratings, matchups[i][j], i, j, b);
    }
  }
  return log_likelihood;
}

double compute_gradient_matchup(const vector<double> &ratings,
                                const vector<int> &matchup, int p1, int p2,
                                double b, float dL) {
  double linear = compute_linear(ratings, p1, p2, b);
  double s = matchup[0] + (double)matchup[1] / 2;
  int n = matchup[0] + matchup[1] + matchup[2];
  return s * dL - (double)n * dL * exp(linear) / (1 + exp(linear));
}

double compute_gradient_r(const vector<double> &ratings,
                          const vector<vector<vector<int>>> &matchups, int p,
                          double b) {
  double gradient = 0;
  for (size_t i = 0; i < matchups[p].size(); ++i) {
    gradient += compute_gradient_matchup(ratings, matchups[p][i], p, i, b, 1);
  }
  for (size_t i = 0; i < matchups.size(); ++i) {
    if (i == p)
      continue;
    gradient += compute_gradient_matchup(ratings, matchups[i][p], i, p, b, -1);
  }
  return gradient;
}

double compute_gradient_b(const vector<double> &ratings,
                          const vector<vector<vector<int>>> &matchups,
                          double b) {
  double gradient = 0;
  for (size_t i = 0; i < matchups.size(); ++i) {
    for (size_t j = 0; j < matchups[i].size(); ++j) {
      if (i == j)
        continue;
      gradient +=
          compute_gradient_matchup(ratings, matchups[i][j], i, j, b, 1);
    }
  }
  return gradient;
}

double update_ratings(vector<double> &ratings,
                      const vector<vector<vector<int>>> &matchups, double &b,
                      double lr) {
  vector<double> r_grad(ratings.size());
  for (size_t i = 0; i < ratings.size() - 1; ++i) {
    r_grad[i] = compute_gradient_r(ratings, matchups, i, b);
  }
  double b_grad = compute_gradient_b(ratings, matchups, b);
  for (size_t i = 0; i < ratings.size() - 1; ++i) {
    ratings[i] += lr * r_grad[i];
  }
  b += lr * b_grad;

  double ll = compute_log_likelihood(ratings, matchups, b);

  return ll;
}

int main() {
  vector<double> ratings(97, 0);
  std::ifstream fin("ratings.txt");
  for (size_t i = 0; i < ratings.size(); ++i) {
    fin >> ratings[i];
    ratings[i] *= log(10) / 400;
  }
  fin.close();
  vector<vector<vector<int>>> matchups(
      97, vector<vector<int>>(97, vector<int>(3, 0)));
  std::ifstream fin2("results.txt");
  int p1, p2, w, d, l;
  while (fin2 >> p1 >> p2 >> w >> d >> l) {
    matchups[p1][p2][0] = w;
    matchups[p1][p2][1] = d;
    matchups[p1][p2][2] = l;
  }
  fin2.close();
  double b = 0;
  double lr = 2e-6;
  double best_ll = update_ratings(ratings, matchups, b, lr);
  int updates_since_best = 0;
  for (int i = 0; i < 1000000; ++i) {
    double ll = update_ratings(ratings, matchups, b, lr);
    if (ll > best_ll) {
      best_ll = ll;
      updates_since_best = 0;
    } else {
      ++updates_since_best;
      if (updates_since_best > 30) {
        lr /= 2;
        updates_since_best = 0;
      }
    }
    if (i % 1000 == 0) {
      printf("Iteration %d\t%f\t%f\t%f\n", i, ll, b / log(10) * 400, lr);
    }
  }
  std::ofstream fout("gd_ratings.txt");
  for (size_t i = 0; i < ratings.size(); ++i) {
    fout << ratings[i] * 400 / log(10) << endl;
  }
  std::ofstream fout2("validation.txt");
  for (size_t i = 0; i < matchups.size(); ++i) {
    for (size_t j = 0; j < matchups[i].size(); ++j) {
      if (i == j)
        continue;
      double actual_score =
          (matchups[i][j][0] + (double)matchups[i][j][1] / 2) /
          (matchups[i][j][0] + matchups[i][j][1] + matchups[i][j][2]);
      fout2 << i << " " << j << " " << ratings[i] / log(10) * 400 << " "
            << ratings[j] / log(10) * 400 << " " << actual_score << " "
            << get_expected_score(ratings, i, j, b) << endl;
    }
  }
  return 0;
}