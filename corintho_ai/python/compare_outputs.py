import numpy as np
import tensorflow as tf


def get_model_tensorflow(path):
    """
    Get a tensorflow model from a path
    """
    model = tf.keras.models.load_model(path)
    return model


def get_model_tflite(path):
    """
    Get a tensorflow lite model from a path
    """
    model = tf.lite.Interpreter(model_path=path)
    print("Model input")
    print(model.get_input_details())
    print("Model output")
    print(model.get_output_details())
    return model


def get_prediction_tensorflow(model, input):
    """
    Get a prediction from a tensorflow model
    """
    return model.predict(input)


def get_prediction_tflite(model, input):
    """
    Get a prediction from a tensorflow lite model
    """
    input = input.astype(np.float32)
    model.allocate_tensors()
    input_details = model.get_input_details()
    output_details = model.get_output_details()
    model.set_tensor(input_details[0]["index"], input)
    model.invoke()
    return model.get_tensor(output_details[0]["index"]), model.get_tensor(
        output_details[1]["index"]
    )


def main():
    """
    Compare tensorflow and tensorflow lite models
    """
    for n in range(93):
        # Get a tensorflow model
        model_tensorflow = get_model_tensorflow(
            f"../../generations/gen_{n}/model"
        )
        # Get a tensorflow lite model
        model_tflite = get_model_tflite(f"./tflite_models/model_{n}.tflite")
        # Input is 70x1
        input = np.array([0] * 64 + [1] * 6).reshape(1, -1)
        # Get predictions from both models
        prediction_tensorflow = get_prediction_tensorflow(
            model_tensorflow, input
        )
        prediction_tflite = get_prediction_tflite(model_tflite, input)
        # Compare predictions
        eval_difference = np.abs(
            prediction_tensorflow[0] - prediction_tflite[1]
        )
        print(f"Eval difference: {eval_difference}")
        prob_difference = prediction_tensorflow[1] - prediction_tflite[0]
        print(f"Prob difference: {max(prob_difference[0])}")


if __name__ == "__main__":
    main()
