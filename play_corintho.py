from play import play
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--turn",
        type=int,
        help="Location of model",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Location of model",
    )
    args = vars(parser.parse_args())
    play(
        model_location=args["model"],
        player_turn=int(args["turn"]),
    )


if __name__ == "__main__":
    main()
