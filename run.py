# Ref: grounded_sam2_hf_model_demo.py
import argparse
import os
import glob
import cv2
import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm  # ✅ NEW: progress bar
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

def process_folder(folder_path, text_prompt, sam2_predictor, processor, grounding_model, device):
    img_files = glob.glob(os.path.join(folder_path, "bdy_*.*"))
    if len(img_files) == 0:
        return
    img_path = img_files[0]

    image = Image.open(img_path).convert("RGB")
    sam2_predictor.set_image(np.array(image))

    inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = grounding_model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids,
        box_threshold=0.4, text_threshold=0.3,
        target_sizes=[image.size[::-1]]
    )

    input_boxes = results[0]["boxes"].cpu().numpy() if len(results) > 0 else np.empty((0, 4))
    preproc_dir = os.path.join(folder_path, "pre_processing")
    os.makedirs(preproc_dir, exist_ok=True)
    mask_output_path = os.path.join(preproc_dir, "black_fg_mask_groundedsam2.png")

    if input_boxes.shape[0] == 0:
        binary_mask = 255 * np.ones((image.height, image.width), dtype=np.uint8)
        cv2.imwrite(mask_output_path, binary_mask)
        return

    masks, _, _ = sam2_predictor.predict(
        point_coords=None, point_labels=None,
        box=input_boxes, multimask_output=False
    )
    if masks.ndim == 4:
        masks = masks.squeeze(1)
    combined_mask = np.any(masks, axis=0)
    binary_mask = np.where(combined_mask, 0, 255).astype(np.uint8)
    cv2.imwrite(mask_output_path, binary_mask)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, default="/home/shenzhen/Datasets/dataset_with_garment")
    parser.add_argument("--grounding-model", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--text-prompt", default="person.")
    parser.add_argument("--sam2-checkpoint", default="./checkpoints/sam2.1_hiera_large.pt")
    parser.add_argument("--sam2-model-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--force-cpu", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu"

    sam2_model = build_sam2(args.sam2_model_config, args.sam2_checkpoint, device=device)
    sam2_predictor = SAM2ImagePredictor(sam2_model)
    model_id = args.grounding_model
    processor = AutoProcessor.from_pretrained(model_id)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    torch.autocast(device_type=device, dtype=torch.bfloat16).__enter__()
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # ✅ tqdm progress bar
    subfolders = sorted([f for f in os.listdir(args.input_dir) if os.path.isdir(os.path.join(args.input_dir, f))])
    for subfolder in tqdm(subfolders, desc="Processing Folders", unit="folder"):
        folder_path = os.path.join(args.input_dir, subfolder)
        process_folder(folder_path, args.text_prompt, sam2_predictor, processor, grounding_model, device)


if __name__ == "__main__":
    main()