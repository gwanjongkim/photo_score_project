import os
import csv
from PIL import Image, ImageOps
import numpy as np

failed_image_path = "data/raw/paq2piq/voc_emotic_ava/AVA__325065.jpg"
output_dir = "outputs/musiq_kaggle_failure_diagnosis_20260601"
os.makedirs(output_dir, exist_ok=True)

def inspect_image(path):
    results = {
        "path": path,
        "exists": os.path.exists(path),
        "file_size": os.path.getsize(path) if os.path.exists(path) else None,
        "pil_open_success": False,
        "mode": None,
        "width": None,
        "height": None,
        "aspect_ratio": None,
        "has_exif": False,
        "exif_orientation": None,
        "transpose_changed_dims": False,
        "rgb_conversion_success": False,
        "reencode_success": False,
        "notes": ""
    }
    
    if not results["exists"]:
        return results

    try:
        with Image.open(path) as img:
            results["pil_open_success"] = True
            results["mode"] = img.mode
            results["width"], results["height"] = img.size
            results["aspect_ratio"] = results["width"] / results["height"]
            
            exif = img.getexif()
            if exif:
                results["has_exif"] = True
                results["exif_orientation"] = exif.get(274) # 0x0112 is Orientation
            
            img_transposed = ImageOps.exif_transpose(img)
            if img_transposed.size != img.size:
                results["transpose_changed_dims"] = True
            
            try:
                rgb_img = img_transposed.convert("RGB")
                results["rgb_conversion_success"] = True
                
                temp_reencode = os.path.join(output_dir, "reencoded_test.jpg")
                rgb_img.save(temp_reencode, "JPEG")
                results["reencode_success"] = True
                os.remove(temp_reencode)
            except Exception as e:
                results["notes"] += f"RGB/Reencode error: {str(e)}; "

    except Exception as e:
        results["notes"] += f"PIL error: {str(e)}; "
        
    return results

inspection = inspect_image(failed_image_path)

with open(os.path.join(output_dir, "failed_image_inspection.csv"), "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=inspection.keys())
    writer.writeheader()
    writer.writerow(inspection)

print(f"Inspection saved to {output_dir}/failed_image_inspection.csv")
