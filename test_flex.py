import tensorflow as tf
import numpy as np

# Create a model with tf.roll
class RollModel(tf.Module):
    @tf.function(input_signature=[tf.TensorSpec([1, 10, 10, 1], tf.float32)])
    def __call__(self, x):
        return tf.roll(x, shift=[1, 1], axis=[1, 2])

model = RollModel()
converter = tf.lite.TFLiteConverter.from_concrete_functions([model.__call__.get_concrete_function()])
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model = converter.convert()

interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()
print("Success allocate")
input_data = np.random.randn(1, 10, 10, 1).astype(np.float32)
interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_data)
interpreter.invoke()
print("Success invoke")
