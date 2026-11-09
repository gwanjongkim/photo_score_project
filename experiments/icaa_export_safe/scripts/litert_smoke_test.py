import ai_edge_litert.interpreter as litert
import numpy as np
import sys
import os

def smoke_test(model_path):
    print(f"Testing model with LiteRT: {model_path}")
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    try:
        interpreter = litert.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        print("Model loaded successfully with LiteRT.")
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print("\nInput Details:")
        for i, detail in enumerate(input_details):
            print(f"  Input {i}: {detail['name']}, Shape: {detail['shape']}, Type: {detail['dtype']}")
            
        print("\nOutput Details:")
        for i, detail in enumerate(output_details):
            print(f"  Output {i}: {detail['name']}, Shape: {detail['shape']}, Type: {detail['dtype']}")
            
        # Dummy inference
        input_shape = input_details[0]['shape']
        dummy_input = np.random.randn(*input_shape).astype(input_details[0]['dtype'])
        interpreter.set_tensor(input_details[0]['index'], dummy_input)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        print(f"\nInference successful. Output shape: {output_data.shape}")
        print(f"Output sample: {output_data}")
        
    except Exception as e:
        print(f"\nError during LiteRT smoke test: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python litert_smoke_test.py <model_path>")
    else:
        smoke_test(sys.argv[1])
