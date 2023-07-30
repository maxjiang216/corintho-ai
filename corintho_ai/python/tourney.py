from tourney import run


def main():
    print("Running tourney.py")
    run(
        ["tflite_model.tflite", "tflite_model.tflite"],
        "test_games.txt",
        1,
        "test_logs",
    )
    print("Done running tourney.py")


if __name__ == "__main__":
    main()
