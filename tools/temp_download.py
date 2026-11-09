
import os
import sys
from huggingface_hub import snapshot_download
from pathlib import Path

def main():
    """
    Downloads the VILA-1.5-3B model from Hugging Face Hub.
    """
    model_id = "Efficient-Large-Model/VILA1.5-3b"
    target_dir = Path(__file__).resolve().parents[1] / "checkpoints" / "VILA1.5-3b"
    
    print(f"Target directory: {target_dir}")
    target_dir.mkdir(exist_ok=True)

    print(f"Starting download of model {model_id} to {target_dir}...")
    print("This may require a Hugging Face login. If it fails, run 'huggingface-cli login'.")
    
    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
            token=True,
        )
        print("Download completed successfully.")
    except Exception as e:
        print(f"An error occurred during download: {e}", file=sys.stderr)
        print("
---", file=sys.stderr)
        print("Download failed. This is likely due to one of two reasons:", file=sys.stderr)
        print("1. You are not logged into Hugging Face.", file=sys.stderr)
        print("2. You have not accepted the license terms for a base model (like Llama 3).", file=sys.stderr)
        print("
ACTION: Please log in via 'huggingface-cli login' and accept any required licenses on the model page, then try again.", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
