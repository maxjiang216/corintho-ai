import argparse
import heapq
import os
from multiprocessing import Pool, cpu_count

from tourney import run


def run_helper(args):
    """
    Run a single match
    """

    (
        model_paths,
        players_file,
        match_file,
        log_folder,
        num_threads,
        folder,
        id,
    ) = args

    run(model_paths, players_file, match_file, log_folder, num_threads)

    with open(os.path.join(folder, f"done_{id}.txt"), "w") as f:
        f.write(match_file)


def get_args():
    """
    Read command line arguments
    """
    # Parse flags
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder",
        type=str,
        default="",
        help="Path to folder containing results",
    )
    parser.add_argument(
        "--num_games",
        type=int,
        default=10000,
        help="Number of games to play",
    )
    parser.add_argument(
        "--num_logged",
        type=int,
        default=10,
        help="Number of games to log",
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=1,
        help="Number of threads to use",
    )

    args = vars(parser.parse_args())

    if args["num_threads"] < 1:
        args["num_threads"] = max(1, cpu_count() - 8)

    return args


def get_models(folder):
    """
    Read model paths from filename
    """

    with open(os.path.join(folder, "models.txt"), "r") as f:
        model_paths = f.readlines()
        model_paths = [path.strip() for path in model_paths]

    return model_paths


def get_variance(n1, m1, n2, m2):
    """
    Calculate variance of score distribution
    """

    return (n1 + 1) * (m1 + 1) / ((n1 + m1 + 2) ** 2 * (n1 + m1 + 3)) + (
        n2 + 1
    ) * (m2 + 1) / ((n2 + m2 + 2) ** 2 * (n2 + m2 + 3))


def write_games(
    num_players, num_games, folder, round_folder, num_logged=10, num_threads=1
):
    """
    Collects all game results from result_folder
    Chooses num_games matches based on score variance
    """

    results = []
    for item in os.listdir(folder):
        item_path = os.path.join(folder, item)
        if os.path.isdir(item_path):
            scores_file_path = os.path.join(item_path, "results.txt")

            if os.path.exists(scores_file_path):
                with open(scores_file_path, "r") as f:
                    scores = f.readlines()
                    scores = [score.split() for score in scores]
                    scores = [
                        ((int(score[0]), int(score[1])), float(score[2]))
                        for score in scores
                    ]
                results.extend(scores)

    scores = {}
    for result in results:
        if result[0] not in scores:
            # Wins, Loosses
            scores[result[0]] = [0, 0]
        if result[1] == 0:
            scores[result[0]][1] += 1
        elif result[1] == 0.5:
            scores[result[0]][0] += 0.5
            scores[result[0]][1] += 0.5
        else:
            scores[result[0]][0] += 1
    with open(os.path.join(round_folder, "scores_list.txt"), "w+") as f:
        f.write(f"{scores}\n")

    variances = {}

    for player1 in range(num_players):
        for player2 in range(player1 + 1, num_players):
            pair = (player1, player2)
            if pair in variances:
                continue
            reverse_pair = (player2, player1)
            if pair in scores:
                n1 = scores[pair][0]
                m1 = scores[pair][1]
            else:
                n1 = 0
                m1 = 0
            if reverse_pair in scores:
                n2 = scores[reverse_pair][0]
                m2 = scores[reverse_pair][1]
            else:
                n2 = 0
                m2 = 0
            variances[pair] = (
                get_variance(n1, m1, n2, m2),
                n1,
                m1,
                n2,
                m2,
            )
    with open(os.path.join(round_folder, "variances_dict.txt"), "w+") as f:
        f.write(f"{variances}\n")

    variances = [
        (
            -1 * variances[key][0],
            (
                key,
                (
                    variances[key][1],
                    variances[key][2],
                    variances[key][3],
                    variances[key][4],
                ),
            ),
        )
        for key in variances
    ]
    with open(os.path.join(round_folder, "variances.txt"), "w+") as f:
        for variance in variances:
            f.write(f"{variance}\n")
    # Setup heap
    heapq.heapify(variances)

    matches = []
    # Get matches
    for _ in range(num_games):
        _, item = heapq.heappop(variances)
        matches.append(item[0])
        n1, m1, n2, m2 = item[1]
        # Assume expected result to decrease variance of chosen pairs
        n1 += n1 / (n1 + m1) if n1 + m1 > 0 else 0.5
        m1 += m1 / (n1 + m1) if n1 + m1 > 0 else 0.5
        n2 += n2 / (n2 + m2) if n2 + m2 > 0 else 0.5
        m2 += m2 / (n2 + m2) if n2 + m2 > 0 else 0.5
        heapq.heappush(
            variances,
            (-1 * get_variance(n1, m1, n2, m2), (item[0], (n1, m1, n2, m2))),
        )

    # Write matches into file
    for i in range(num_threads):
        match_file = os.path.join(round_folder, f"matches_{i}.txt")
        with open(match_file, "w+") as f:
            f.write(f"{len(matches[i::num_threads])}\n")
            for match in matches[i::num_threads]:
                # Always play a match
                f.write(f"{match[0]}\t{match[1]}\t0\n")
                f.write(
                    f"{match[1]}\t{match[0]}\t0\n"
                )


def combine_results(folder, num_threads):
    """
    Combine results from multiple threads
    """

    results = []
    for i in range(num_threads):
        with open(os.path.join(folder, f"logs_{i}", "scores.txt"), "r") as f:
            results.extend(f.readlines())

    with open(os.path.join(folder, "results.txt"), "w+") as f:
        f.write("".join(results))

def get_games(folder):
    """
    Return a list of games and results
    """
    games = {}
    for item in os.listdir(folder):
        item_path = os.path.join(folder, item)
        if os.path.isdir(item_path):
            scores_file_path = os.path.join(item_path, "results.txt")

            if os.path.exists(scores_file_path):
                with open(scores_file_path, "r") as f:
                    scores = f.readlines()
                    scores = [score.split() for score in scores]
                    scores = [
                        ((int(score[0]), int(score[1])), float(score[2]))
                        for score in scores
                    ]
                for score in scores:
                    if score[0][0] not in games:
                        games[score[0][0]] = []
                    games[score[0][0]].append((score[0][1], score[1]))
                    if score[0][1] not in games:
                        games[score[0][1]] = []
                    games[score[0][1]].append((score[0][0], 1 - score[1]))

    return games

def get_matchups(round_folder, folder):
    """
    Get the matchup score for each pairing
    """
    games = get_games(folder)
    matchups = {}
    for player in games:
        matchups[player] = {}
        for game in games[player]:
            if game[0] not in matchups[player]:
                matchups[player][game[0]] = [0, 0]
            matchups[player][game[0]][0] += game[1]
            matchups[player][game[0]][1] += 1

    with open(os.path.join(round_folder, "matchups.txt"), "w+") as f:
        for player in matchups:
            for opponent in matchups[player]:
                f.write(
                    f"{player}\t{opponent}\t{matchups[player][opponent][0]} / {matchups[player][opponent][1]} = {matchups[player][opponent][0] / matchups[player][opponent][1]}\n"
                )

def get_performance(score, opponents):
    """
    Use binary search to find performance rating
    """

    # Find performance rating
    lower_bound = -10000
    upper_bound = 20000
    while upper_bound - lower_bound > 0.1:
        mid = (upper_bound + lower_bound) / 2
        expected_score = sum(
            [
                1 / (1 + 10 ** ((opponent - mid) / 400))
                for opponent in opponents
            ]
        ) / len(opponents)
        if expected_score < score:
            lower_bound = mid
        else:
            upper_bound = mid

    return lower_bound


def get_performance_ratings(games, players, round_folder):
    """
    Compute performance ratings
    """

    new_players = {}
    delta = 0

    for player, rating in players.items():
        if rating[1]:
            new_players[player] = (0, True)
        elif player not in games:
            new_players[player] = (rating[0], False)
        else:
            opponents = [players[opponent[0]][0] for opponent in games[player]]
            score = sum([game[1] for game in games[player]])
            if len(opponents) == 0:
                new_rating = rating[0]
            elif score == 0:
                new_rating = 0
            else:
                if score == len(games[player]):
                    score -= 0.25  # Prevent perfect score
                new_rating = get_performance(score / len(opponents), opponents)
                with open(os.path.join(round_folder, "performance.txt"), "a+") as f:
                    f.write(
                        f"{player}\t{rating[0]}\t{new_rating}\t{score}/{len(opponents)}\t{sum(opponents)/len(opponents)}\n"
                    )
            new_players[player] = (new_rating, False)
            delta = max(delta, abs(new_rating - rating[0]))

    return new_players, delta


def write_ratings(folder, round_folder):
    """
    Compute and write new ratings
    Iteratively find performance rating
    Random players are fixed at a rating of 0.
    """

    # Find seed ratings
    with open(os.path.join(folder, "ratings.txt"), "r") as f:
        ratings = f.readlines()
        ratings = [float(rating.strip()) for rating in ratings]
    # Find random players
    with open(os.path.join(folder, "players.txt"), "r") as f:
        players = f.readlines()[1:]
        players = [player.strip().split() for player in players]
        with open(os.path.join(round_folder, "get_games.txt"), "a+") as f:
            f.write(f"{players}\n")
        randomness = [player[-1] == "1" for player in players]
        with open(os.path.join(round_folder, "get_games.txt"), "a+") as f:
            f.write(f"{randomness}\n")
    players = {}
    for i, rating in enumerate(ratings):
        if randomness[i]:
            players[i] = (0, True)
        else:
            players[i] = (rating, False)

    # Get games
    games = get_games(folder)
    # Write matchups
    get_matchups(round_folder, folder)

    # Compute performance ratings
    delta = 999999
    while delta > 1:
        players, delta = get_performance_ratings(games, players, round_folder)
        with open(os.path.join(round_folder, "performance.txt"), "a+") as f:
            f.write(f"{players}\n{delta}\n")

    # Write new ratings
    out_string = "\n".join([f"{players[player][0]}" for player in players])
    with open(os.path.join(round_folder, "ratings.txt"), "w+") as f:
        f.write(out_string)

    with open(
        os.path.join(folder, "ratings.txt"),
        "w+",
    ) as f:
        f.write(out_string)


def main():
    args = get_args()

    model_paths = get_models(args["folder"])

    current_round = int(
        open(os.path.join(args["folder"], "current_round.txt"), "r")
        .read()
        .strip()
    )

    with open(os.path.join(args["folder"], "players.txt"), "r") as f:
        num_players = len(f.readlines()) - 1
    if not os.path.exists(
        os.path.join(args["folder"], f"round_{current_round}")
    ):
        os.mkdir(os.path.join(args["folder"], f"round_{current_round}"))
    round_folder = os.path.join(args["folder"], f"round_{current_round}")
    if not os.path.exists(round_folder):
        os.mkdir(round_folder)
    write_games(
        num_players,
        args["num_games"],
        args["folder"],
        round_folder,
        args["num_logged"],
        args["num_threads"],
    )

    args_list = []
    for i in range(args["num_threads"]):
        if not os.path.exists(os.path.join(round_folder, f"logs_{i}")):
            os.mkdir(os.path.join(round_folder, f"logs_{i}"))
        args_list.append(
            (
                model_paths,
                os.path.join(args["folder"], "players.txt"),
                os.path.join(
                    args["folder"],
                    f"round_{current_round}",
                    f"matches_{i}.txt",
                ),
                os.path.join(
                    args["folder"],
                    f"round_{current_round}",
                    f"logs_{i}",
                ),
                1,
                os.path.join(
                    args["folder"],
                    f"round_{current_round}",
                ),
                i,
            )
        )

    with Pool(args["num_threads"]) as pool:
        pool.map(run_helper, args_list)

    # Combine results
    combine_results(
        os.path.join(args["folder"], f"round_{current_round}"),
        args["num_threads"],
    )

    # Compute new ratings
    write_ratings(
        args["folder"],
        os.path.join(args["folder"], f"round_{current_round}"),
    )

    # Update current round
    with open(
        os.path.join(args["folder"], "current_round.txt"),
        "w",
    ) as f:
        f.write(str(current_round + 1))


if __name__ == "__main__":
    main()
