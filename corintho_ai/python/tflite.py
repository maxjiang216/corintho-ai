def main():

    import tensorflow as tf

    # Load your Keras model
    keras_model = tf.keras.models.load_model('../generations/gen_78/model')

    # Convert the Keras model to TensorFlow Lite format
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    #converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    # Save the TensorFlow Lite model
    with open('tflite_model.tflite', 'wb') as f:
        f.write(tflite_model)

if __name__ == '__main__':
    main()