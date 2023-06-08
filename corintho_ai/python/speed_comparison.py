def main():

    import numpy as np
    import tensorflow as tf
    import time

    # Load your Keras model
    keras_model = tf.keras.models.load_model('../generations/gen_78/model')

    # Load your TensorFlow Lite model
    tflite_model_path = 'tflite_model.tflite'
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path, num_threads=2)
    interpreter.allocate_tensors()

    # Load your TensorFlow Lite model
    tflite_quant_model_path = 'tflite_model_quant.tflite'
    interpreter_quant = tf.lite.Interpreter(model_path=tflite_quant_model_path, num_threads=2)
    interpreter_quant.allocate_tensors()

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Generate random input data
    input_shape = list(input_details[0]['shape'])
    input_shape[0] = 1300 # Set batch size to 300
    input_data = np.random.random_sample(input_shape).astype(np.float32)

    # Time Keras model prediction
    start_time = time.perf_counter()
    keras_output = keras_model.predict(input_data, batch_size=input_shape[0])
    keras_time = time.perf_counter() - start_time

    # Time TensorFlow Lite model prediction
    interpreter.resize_tensor_input(input_details[0]['index'], input_shape)
    interpreter.allocate_tensors()
    interpreter.set_tensor(input_details[0]['index'], input_data)
    start_time = time.perf_counter()
    interpreter.invoke()
    tflite_output = interpreter.get_tensor(output_details[1]['index'])
    tflite_time = time.perf_counter() - start_time

    # Time TensorFlow Lite model prediction
    interpreter_quant.resize_tensor_input(input_details[0]['index'], input_shape)
    interpreter_quant.allocate_tensors()
    interpreter_quant.set_tensor(input_details[0]['index'], input_data)
    start_time = time.perf_counter()
    interpreter_quant.invoke()
    tflite_quant_output = interpreter_quant.get_tensor(output_details[1]['index'])
    tflite_quant_time = time.perf_counter() - start_time

    #print(f"Keras output: {keras_output[0]}")
    #print(f"TensorFlow Lite output: {tflite_output}")

    print(f"Keras model prediction time: {keras_time:.6f} seconds")
    print(f"TensorFlow Lite model prediction time: {tflite_time:.6f} seconds")
    print(f"TensorFlow Lite quant model prediction time: {tflite_quant_time:.6f} seconds")
    print(f"Speed improvement: {keras_time / tflite_time:.2f} times")
    print(f"Speed improvement: {tflite_time / tflite_quant_time:.2f} times")

if __name__ == '__main__':
    main()
