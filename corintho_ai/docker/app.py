from flask import Flask, jsonify, request
from flask_cors import CORS
from play_corintho import choose_move

app = Flask(__name__)
CORS(app)


@app.route("/choose_move", methods=["POST"])
def choose_move_route():
    data = request.get_json()
    return jsonify(
        choose_move(
            data["gameState"],
            data["timeLimit"],
            data["searchesPerEval"],
            data["maxNodes"],
        )
    )


@app.route("/warm_up", methods=["POST"])
def warm_up():
    return jsonify({"status": 200})


if __name__ == "__main__":
    app.run(debug=True)
