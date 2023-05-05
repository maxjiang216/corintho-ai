def main():

    import numpy as np
    import tensorflow as tf
    import time

    # Load your Keras model
    keras_model = tf.keras.models.load_model('../generations/gen_78/model')

    # Load your TensorFlow Lite model
    tflite_model_path = '../tflite_model.tflite'
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Generate random input data
    input_shape = list(input_details[0]['shape'])
    input_shape[0] = 1600  # Set batch size to 300
    input_data = np.random.random_sample(input_shape).astype(np.float32)

    # Time Keras model prediction
    start_time = time.time()
    keras_output = keras_model.predict(input_data)
    keras_time = time.time() - start_time

    # Time TensorFlow Lite model prediction
    interpreter.resize_tensor_input(input_details[0]['index'], input_shape)
    interpreter.allocate_tensors()
    interpreter.set_tensor(input_details[0]['index'], input_data)
    start_time = time.time()
    interpreter.invoke()
    tflite_output = interpreter.get_tensor(output_details[0]['index'])
    tflite_time = time.time() - start_time

    print(f"Keras model prediction time: {keras_time:.6f} seconds")
    print(f"TensorFlow Lite model prediction time: {tflite_time:.6f} seconds")
    print(f"Speed improvement: {keras_time / tflite_time:.2f} times")

if __name__ == '__main__':
    main()
