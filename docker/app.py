from flask import Flask, request, jsonify
from play_corintho import choose_move
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/choose_move", methods=["POST"])
def choose_move():
    data = request.get_json()
    return jsonify(
        choose_move(
            data["gameState"],
            data["timeLimit"],
            data["searchesPerEval"],
            data["maxNodes"],
        )
    )


if __name__ == "__main__":
    app.run(debug=True)
