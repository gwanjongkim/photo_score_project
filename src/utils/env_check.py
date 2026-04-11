
import tensorflow as tf

print(f"TensorFlow Version: {tf.__version__}")
print(f"Available GPUs: {tf.config.list_physical_devices('GPU')}")
