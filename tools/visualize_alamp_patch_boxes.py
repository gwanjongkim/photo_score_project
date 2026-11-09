"""A-LAMP-paper-MPNet patch box visualization utility."""
import argparse
import json
import logging
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def main():
    parser = argparse.ArgumentParser(description="Visualize A-LAMP patch boxes.")
    parser.add_argument("--input_jsonl", required=True, help="Path to patch boxes JSONL file.")
    parser.add_argument("--output_dir", required=True, help="Directory to save visualization images.")
    parser.add_argument("--max_images", type=int, default=30, help="Maximum number of images to visualize.")
    parser.add_argument("--start_index", type=int, default=0, help="Start index in JSONL file.")
    args = parser.parse_args()

    input_path = Path(args.input_jsonl)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        logging.error(f"Input file not found: {input_path}")
        return

    processed_count = 0
    skipped_count = 0
    current_line_idx = 0

    # Try to load a font, fallback to default
    try:
        # Common linux font path
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except Exception:
        font = ImageFont.load_default()

    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            if processed_count >= args.max_images:
                break
            
            line_idx = current_line_idx
            current_line_idx += 1
            
            if line_idx < args.start_index:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                logging.warning(f"Failed to parse JSON line: {e}")
                continue

            if data.get("skipped"):
                skipped_count += 1
                continue

            image_path_str = data.get("resolved_image_path") or data.get("image_path")
            if not image_path_str:
                logging.warning("No image path found in record.")
                skipped_count += 1
                continue

            image_path = Path(image_path_str)
            if not image_path.exists():
                logging.warning(f"Image not found: {image_path}")
                skipped_count += 1
                continue

            try:
                with Image.open(image_path) as img:
                    img = img.convert("RGB")
                    draw = ImageDraw.Draw(img)
                    
                    boxes = data.get("boxes_abs_xyxy", [])
                    scores = data.get("patch_scores", [])
                    
                    for i, box in enumerate(boxes):
                        x1, y1, x2, y2 = box
                        color = (255, 0, 0) # Red
                        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                        
                        # Draw label
                        label = f"{i + 1}"
                        if i < len(scores):
                            label += f" ({scores[i]:.3f})"
                        
                        # Draw text background for readability
                        text_pos = (x1 + 5, y1 + 5)
                        draw.text(text_pos, label, fill=color, font=font)

                    save_path = output_dir / f"viz_{image_path.name}"
                    img.save(save_path)
                    processed_count += 1
                    
            except Exception as e:
                logging.error(f"Error processing image {image_path}: {e}")
                skipped_count += 1
                continue

    logging.info(f"Visualization complete.")
    logging.info(f"Processed: {processed_count}")
    logging.info(f"Skipped/Failed: {skipped_count}")
    logging.info(f"Output directory: {output_dir}")

if __name__ == "__main__":
    main()
