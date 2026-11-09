import torch
print(f"torch_version={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device_count={torch.cuda.device_count()}")
    print(f"device_name={torch.cuda.get_device_name(0)}")
