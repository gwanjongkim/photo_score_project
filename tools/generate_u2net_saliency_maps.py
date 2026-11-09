import os
import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.autograd import Variable
from torchvision import transforms
from PIL import Image

from src.saliency.u2net_model import U2NET

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def _resolve_image_paths(frame, image_col, image_dir):
    image_root = Path(image_dir) if image_dir else None
    def _resolve_path(value):
        raw = Path(str(value))
        if raw.is_absolute() or raw.exists():
            return str(raw)
        if image_root is not None:
            candidate = image_root / raw.name
            if candidate.exists():
                return str(candidate)
        return str(raw)
    return frame[image_col].map(_resolve_path)

def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d - mi) / (ma - mi + 1e-8)
    return dn

def save_output(image_name, pred, d_dir):
    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np * 255).convert('L')
    img_name = Path(image_name).stem
    save_path = Path(d_dir) / (img_name + '.png')
    im.save(save_path)
    return str(save_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--image_dir")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--weights", default="models/saliency/u2net.pth")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Load model
    logging.info(f"Loading U2NET from {args.weights}")
    net = U2NET(3, 1)
    if args.device == "cuda":
        net.load_state_dict(torch.load(args.weights))
        net.cuda()
    else:
        net.load_state_dict(torch.load(args.weights, map_location="cpu"))
    net.eval()

    df = pd.read_csv(args.csv)
    df["_resolved_path"] = _resolve_image_paths(df, "image_path", args.image_dir)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metadata_path = output_dir / "saliency_metadata.jsonl"
    
    transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    skipped = 0
    with open(metadata_path, "w") as meta_f:
        for idx, row in df.iterrows():
            img_path = row["_resolved_path"]
            if not Path(img_path).exists():
                logging.warning(f"Image not found: {img_path}")
                skipped += 1
                continue
                
            try:
                img = Image.open(img_path).convert('RGB')
                w, h = img.size
                inputs_test = transform(img)
                inputs_test = inputs_test.unsqueeze(0)
                if args.device == "cuda":
                    inputs_test = inputs_test.cuda()

                with torch.no_grad():
                    d1, d2, d3, d4, d5, d6, d7 = net(inputs_test)

                # normalization
                pred = d1[:, 0, :, :]
                pred = normPRED(pred)
                
                # Resize pred back to original size
                pred_resized = F.interpolate(pred.unsqueeze(1), size=(h, w), mode='bilinear', align_corners=False)
                
                mask_path = save_output(img_path, pred_resized, output_dir)
                
                record = {
                    "image_path": str(row["image_path"]),
                    "saliency_map_path": mask_path,
                    "width": w,
                    "height": h
                }
                meta_f.write(json.dumps(record) + "\n")
                
            except Exception as e:
                logging.error(f"Error processing {img_path}: {e}")
                skipped += 1
                
            if (idx + 1) % 100 == 0:
                logging.info(f"Processed {idx+1} images...")

    logging.info(f"Done. Generated masks for {len(df) - skipped} images. Skipped: {skipped}")

if __name__ == "__main__":
    import pandas as pd
    import torch.nn.functional as F
    main()
