import onnx
import sys
from collections import Counter
from pathlib import Path

def inspect_onnx(model_path):
    if not Path(model_path).exists():
        print(f"Error: Model not found at {model_path}")
        return

    model = onnx.load(model_path)
    graph = model.graph

    print(f"Model Path: {model_path}")
    
    # Inputs
    print("\nInputs:")
    for input in graph.input:
        shape = []
        for dim in input.type.tensor_type.shape.dim:
            shape.append(dim.dim_value if dim.dim_value > 0 else "dynamic")
        print(f"  Name: {input.name}, Shape: {shape}")

    # Outputs
    print("\nOutputs:")
    for output in graph.output:
        shape = []
        for dim in output.type.tensor_type.shape.dim:
            shape.append(dim.dim_value if dim.dim_value > 0 else "dynamic")
        print(f"  Name: {output.name}, Shape: {shape}")

    # Operator Histogram
    ops = [node.op_type for node in graph.node]
    op_counts = Counter(ops)
    
    print("\nOperator Histogram:")
    for op, count in sorted(op_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {op}: {count}")

    # Check for risky ops
    risky_ops = ["GridSample", "Einsum", "Gather", "Scatter", "Resize"]
    print("\nRisky Operators Check:")
    for op in risky_ops:
        if op in op_counts:
            print(f"  {op}: FOUND ({op_counts[op]})")
        else:
            print(f"  {op}: Not found")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_onnx.py <model_path>")
    else:
        inspect_onnx(sys.argv[1])
