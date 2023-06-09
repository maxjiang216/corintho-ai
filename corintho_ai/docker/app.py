from flask import Flask, request, jsonify
from play_corintho import choose_move
from flask_cors import CORS
import tensorflow as tf

app = Flask(__name__)
CORS(app)

# Load the TFLite model
tflite_model_path = "tflite_model.tflite"
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()


@app.route("/choose_move", methods=["POST"])
def choose_move_route():
    data = request.get_json()
    return jsonify(
        choose_move(
            data["gameState"],
            data["timeLimit"],
            data["searchesPerEval"],
            data["maxNodes"],
            interpreter,
        )
    )


@app.route("/warm_up", methods=["POST"])
def warm_up():
    return jsonify({"status": 200})


if __name__ == "__main__":
    app.run(debug=True)
