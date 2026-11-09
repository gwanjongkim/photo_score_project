import tensorflow as tf
import numpy as np
import os

# Create a model with tf.roll
class RollModel(tf.Module):
    @tf.function(input_signature=[tf.TensorSpec([1, 10, 10, 1], tf.float32)])
    def __call__(self, x):
        return tf.roll(x, shift=[1, 1], axis=[1, 2])

model = RollModel()
converter = tf.lite.TFLiteConverter.from_concrete_functions([model.__call__.get_concrete_function()])
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model = converter.convert()

with open("test.tflite", "wb") as f:
    f.write(tflite_model)

print("Saved to test.tflite")

# New process or same? Let's try same first.
interpreter = tf.lite.Interpreter(model_path="test.tflite")
interpreter.allocate_tensors()
print("Success allocate")
input_data = np.random.randn(1, 10, 10, 1).astype(np.float32)
interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_data)
interpreter.invoke()
print("Success invoke")
