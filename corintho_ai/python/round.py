import argparse
import heapq
import os

from tourney import run


def get_args():
    """
    Read command line arguments
    """
    # Parse flags
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_file",
        type=str,
        default="",
        help="Path to file containing model paths",
    )
    parser.add_argument(
        "--result_folder",
        type=str,
        default="",
        help="Path to folder containing results",
    )
    parser.add_argument(
        "--match_file",
        type=str,
        default="",
        help="Path to file containing matches",
    )
    parser.add_argument(
        "--player_file",
        type=str,
        default="",
        help="Path to file containing player paths",
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
    parser.add_argument(
        "--log_folder",
        type=str,
        default="",
        help="Path to folder for logging",
    )

    return vars(parser.parse_args())


def get_models(model_file):
    """
    Read model paths from filename
    """

    with open(model_file, "r") as f:
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


def write_games(num_games, result_folder, match_file, num_logged=10):
    """
    Collects all game results from result_folder
    Chooses num_games matches based on score variance
    """

    results = []
    for item in os.listdir(result_folder):
        item_path = os.path.join(result_folder, item)
        if os.path.isdir(item_path):
            scores_file_path = os.path.join(item_path, "scores.txt")

            if os.path.exists(scores_file_path):
                with open(scores_file_path, "r") as f:
                    scores = f.readlines()
                    scores = [score.split("\t") for score in scores]
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

    variances = {}

    for key in scores:
        pair = tuple(sorted(key))
        reverse_pair = tuple(sorted(key, reverse=True))
        if pair in variances:
            continue
        n1 = scores[pair][0]
        n2 = scores[reverse_pair][0]
        m1 = scores[pair][1]
        m2 = scores[reverse_pair][1]
        variances[pair] = (
            get_variance(n1, m1, n2, m2),
            n1,
            m1,
            n2,
            m2,
        )

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
    # Setup heap
    heapq.heapify(variances)

    matches = []
    # Get matches
    for _ in range(num_games):
        _, item = heapq.heappop(variances)
        matches.append(item[0])
        n1, m1, n2, m2 = item[1]
        # Assume expected result to decrease variance of chosen pairs
        n1 += n1 / (n1 + m1)
        m1 += m1 / (n1 + m1)
        n2 += n2 / (n2 + m2)
        m2 += m2 / (n2 + m2)
        heapq.push(
            (item[0], (n1, m1, n2, m2)),
            get_variance(n1, m1, n2, m2),
        )

    # Write matches into file
    with open(match_file, "w") as f:
        f.write(f"{len(matches)}\n")
        for match in matches:
            f.write(f"{match[0]}\t{match[1]}\t{1 if num_logged > 0 else 0}\n")
            num_logged -= 1


def main():
    args = get_args()

    model_paths = get_models(args["model_file"])

    write_games(
        args["num_games"],
        args["result_folder"],
        args["match_file"],
        args["num_logged"],
    )

    print("Running tourney.py")
    run(
        model_paths,
        args["player_file"],
        args["match_file"],
        args["num_threads"],
        args["log_folder"],
    )
    print("Done running tourney.py")

    # Compute new ratings


if __name__ == "__main__":
    main()
