import tensorflow as tf
import json

try:
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        details = [tf.config.experimental.get_device_details(gpu) for gpu in gpus]
        print(json.dumps({"gpu_available": True, "gpus_found": len(gpus), "details": details}, indent=2))
    else:
        print(json.dumps({"gpu_available": False, "gpus_found": 0, "details": []}, indent=2))
except Exception as e:
    print(json.dumps({"gpu_available": False, "error": str(e)}, indent=2))
