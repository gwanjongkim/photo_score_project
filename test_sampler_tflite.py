import tensorflow as tf
import numpy as np

def manual_bilinear_grid_sample_nhwc(inputs, grid):
    inputs = tf.convert_to_tensor(inputs)
    grid = tf.convert_to_tensor(grid, dtype=inputs.dtype)

    input_shape = tf.shape(inputs)
    n = input_shape[0]
    h_in = input_shape[1]
    w_in = input_shape[2]
    
    grid_shape = tf.shape(grid)
    h_out = grid_shape[1]
    w_out = grid_shape[2]

    grid_split = tf.split(grid, 2, axis=-1)
    x_norm = tf.squeeze(grid_split[0], axis=-1)
    y_norm = tf.squeeze(grid_split[1], axis=-1)

    h_in_f = tf.cast(h_in, inputs.dtype)
    w_in_f = tf.cast(w_in, inputs.dtype)
    x = (x_norm + 1.0) * 0.5 * (w_in_f - 1.0)
    y = (y_norm + 1.0) * 0.5 * (h_in_f - 1.0)

    x0 = tf.floor(x)
    y0 = tf.floor(y)
    x1 = x0 + 1.0
    y1 = y0 + 1.0

    wx1 = x - x0
    wy1 = y - y0
    wx0 = 1.0 - wx1
    wy0 = 1.0 - wy1

    max_x = w_in_f - 1.0
    max_y = h_in_f - 1.0
    
    valid00 = (x0 >= 0.0) & (x0 <= max_x) & (y0 >= 0.0) & (y0 <= max_y)
    valid01 = (x1 >= 0.0) & (x1 <= max_x) & (y0 >= 0.0) & (y0 <= max_y)
    valid10 = (x0 >= 0.0) & (x0 <= max_x) & (y1 >= 0.0) & (y1 <= max_y)
    valid11 = (x1 >= 0.0) & (x1 <= max_x) & (y1 >= 0.0) & (y1 <= max_y)

    w00 = tf.where(valid00, wy0 * wx0, 0.0)
    w01 = tf.where(valid01, wy0 * wx1, 0.0)
    w10 = tf.where(valid10, wy1 * wx0, 0.0)
    w11 = tf.where(valid11, wy1 * wx1, 0.0)
    
    x0_idx = tf.cast(tf.clip_by_value(x0, 0.0, max_x), tf.int32)
    x1_idx = tf.cast(tf.clip_by_value(x1, 0.0, max_x), tf.int32)
    y0_idx = tf.cast(tf.clip_by_value(y0, 0.0, max_y), tf.int32)
    y1_idx = tf.cast(tf.clip_by_value(y1, 0.0, max_y), tf.int32)

    batch_idx = tf.range(n, dtype=tf.int32)
    batch_idx = tf.reshape(batch_idx, [n, 1, 1])
    batch_idx = tf.tile(batch_idx, [1, h_out, w_out])
    
    def get_val(yi, xi):
        idx = tf.stack([batch_idx, yi, xi], axis=-1)
        return tf.gather_nd(inputs, idx)

    v00 = get_val(y0_idx, x0_idx)
    v01 = get_val(y0_idx, x1_idx)
    v10 = get_val(y1_idx, x0_idx)
    v11 = get_val(y1_idx, x1_idx)
    
    w00 = tf.expand_dims(w00, -1)
    w01 = tf.expand_dims(w01, -1)
    w10 = tf.expand_dims(w10, -1)
    w11 = tf.expand_dims(w11, -1)
    
    return v00 * w00 + v01 * w01 + v10 * w10 + v11 * w11

class TestModel(tf.Module):
    @tf.function(input_signature=[
        tf.TensorSpec([None, 14, 14, 128], tf.float32),
        tf.TensorSpec([None, 196, 196, 2], tf.float32)
    ])
    def __call__(self, inputs, grid):
        return manual_bilinear_grid_sample_nhwc(inputs, grid)

model = TestModel()
converter = tf.lite.TFLiteConverter.from_concrete_functions([model.__call__.get_concrete_function()])
tflite_model = converter.convert()

with open("test_sampler.tflite", "wb") as f:
    f.write(tflite_model)

interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# n=4, h_in=14, w_in=14, c=128
# h_out=196, w_out=196
in_data = np.random.randn(4, 14, 14, 128).astype(np.float32)
grid_data = np.random.uniform(-1, 1, (4, 196, 196, 2)).astype(np.float32)

interpreter.resize_tensor_input(input_details[0]['index'], in_data.shape)
interpreter.resize_tensor_input(input_details[1]['index'], grid_data.shape)
interpreter.allocate_tensors()
interpreter.set_tensor(input_details[0]['index'], in_data)
interpreter.set_tensor(input_details[1]['index'], grid_data)
interpreter.invoke()
out = interpreter.get_tensor(output_details[0]['index'])
print("Success! Output shape:", out.shape)
