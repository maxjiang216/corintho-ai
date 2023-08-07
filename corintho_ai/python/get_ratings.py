import os


def main():
    for i in range(93, -1, -1):
        rating_path = f"../../generations/gen_{i}/rating.txt"

        with open(rating_path, "r") as f:
            rating = f.read().strip()
            print(f"{rating}")


if __name__ == "__main__":
    main()
