import random

import numpy as np


class Rating:
    """
    Holds ratings for players
    Allows GD to be calculated
    """

    def __init__(self, n_players, matchups, lr=1e-6):
        self.n = n_players
        self.matchups = matchups
        self.ratings = [0] * n_players
        self.b = 0
        self.lr = lr
        self.last_ll = -np.inf
        self.updates_since_improvement = 0

    def compute_log_liklihood(self):
        """
        Computes current log liklihood of outcome given ratings and constants
        """

        ll = 0

        for p1 in self.matchups:
            for p2 in self.matchups[p1]:
                ll += self.compute_contribution(p1, p2, self.matchups[p1][p2])

        return ll

    def compute_contribution(self, p1, p2, matchup):
        """
        Compute the contribution of a single matchup to the log liklihood

        matchup: list of [wins, draws, losses]
        """

        l = self.compute_linear(p1, p2)

        return (matchup[0] + matchup[1] / 2) * l - sum(matchup) * np.log(
            1 + np.exp(l)
        )

    def compute_linear(self, p1, p2):
        """
        Compute the linear term
        (r_1-r_2)+b+m(r_1+r_2)
        """

        return (self.ratings[p1] - self.ratings[p2]) + self.b

    def compute_gradient_r(self, k):
        """
        Computes the gradient with respect to player k's rating
        """

        grad = 0

        for opp in self.matchups[k]:
            grad += self.compute_gradient_r_matchup(
                k, opp, self.matchups[k][opp]
            )

        for opp in self.matchups:
            if opp != k:
                grad += self.compute_gradient_r_matchup(
                    opp, k, self.matchups[opp][k], True
                )

        return grad

    def compute_gradient_r_matchup(self, p1, p2, matchup, reverse=False):
        """
        Computes the gcontributions of a single matchup to the gradient of player k's rating
        """

        L = self.compute_linear(p1, p2)
        # Derivative of linear term
        dL = -1 if reverse else 1
        s = matchup[0] + matchup[1] / 2
        n = sum(matchup)

        return s * dL - n * dL * np.exp(L) / (1 + np.exp(L))

    def compute_gradient_b(self):
        """
        Computes the gradient with respect to b
        """

        grad = 0

        for p1 in self.matchups:
            for p2 in self.matchups[p1]:
                grad += self.compute_gradient_b_matchup(
                    p1, p2, self.matchups[p1][p2]
                )

        return grad

    def compute_gradient_b_matchup(self, p1, p2, matchup):
        """
        Computes the contribution of a single matchup to the gradient of b
        """

        L = self.compute_linear(p1, p2)
        s = matchup[0] + matchup[1] / 2
        n = sum(matchup)

        return s - n * np.exp(L) / (1 + np.exp(L))

    def update(self):
        """
        Updates the ratings
        """

        r_gradient = [self.compute_gradient_r(k) for k in range(self.n - 1)]
        b_gradient = self.compute_gradient_b()

        self.ratings = [
            r + self.lr * g for r, g in zip(self.ratings, r_gradient)
        ]
        self.ratings += [0]
        self.b += self.lr * b_gradient

        ll = self.compute_log_liklihood()
        if ll < self.last_ll:
            self.updates_since_improvement += 1
        if self.updates_since_improvement > 20:
            self.lr *= 0.8
            self.updates_since_improvement = 0

        self.last_ll = ll

        print(ll)
        print(self.lr)
        print(self.b * 400 / np.log(10))
        print(1 / (1 + np.exp(self.b)))
        print([r * 400 / np.log(10) for r in self.ratings])
        p1 = random.choice(list(self.matchups.keys()))
        p2 = random.choice(list(self.matchups[p1].keys()))
        matchup = self.matchups[p1][p2]
        print((matchup[0] + matchup[1] / 2) / sum(matchup))
        print(self.ratings[p1] - self.ratings[p2] + self.b)
        print(1 / (1 + np.exp(self.ratings[p1] - self.ratings[p2] + self.b)))


def learn(file):
    """
    Learns ratings from a file
    """

    with open(file, "r") as f:
        lines = f.readlines()
    lines = [line.strip().split() for line in lines]
    lines = [line for line in lines if len(line) == 5]
    lines = [[int(x) for x in line] for line in lines]

    n_players = max([max(line[0], line[1]) for line in lines]) + 1
    matchups = {}
    for line in lines:
        if line[0] not in matchups:
            matchups[line[0]] = {}
        matchups[line[0]][line[1]] = [line[2], line[3], line[4]]

    ratings = Rating(n_players, matchups)

    for i in range(10000):
        ratings.update()


def main():
    learn("results.txt")


if __name__ == "__main__":
    main()
