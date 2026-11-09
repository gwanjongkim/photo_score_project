import tensorflow as tf
import numpy as np
import sys

# Assume test.tflite exists
interpreter = tf.lite.Interpreter(model_path="test.tflite")
interpreter.allocate_tensors()
print("Success allocate")
input_data = np.random.randn(1, 10, 10, 1).astype(np.float32)
interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_data)
interpreter.invoke()
print("Success invoke")
