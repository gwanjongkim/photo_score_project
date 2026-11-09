import tensorflow as tf
import sys

def list_ops(model_path):
    print(f"Listing ops for model: {model_path}")
    try:
        # Load the model and get its details
        with open(model_path, 'rb') as f:
            model_content = f.read()
        
        interpreter = tf.lite.Interpreter(model_content=model_content)
        
        # Get all ops
        ops = []
        for i in range(interpreter._get_ops_details_len()):
            op = interpreter._get_ops_details(i)
            ops.append(op['op_name'])
            
        from collections import Counter
        counts = Counter(ops)
        print("\nTFLite Operator Histogram:")
        for op, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {op}: {count}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python list_tflite_ops.py <model_path>")
    else:
        list_ops(sys.argv[1])
