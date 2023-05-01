import random

def get_random_move(game_state, time_limit):
    return {
        "move": {
            "type": True,
            "pieceType": "base",
            "row": random.randint(0, 4),
            "col": random.randint(0, 4),
        },
        "legalMoves": [],
        "winningMoves": [],
    }