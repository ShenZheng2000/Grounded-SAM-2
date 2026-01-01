# grounded_sam2_flat_folder.py
import argparse
import os
import cv2
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


def process_image(
    img_path,
    out_dir,
    text_prompt,
    sam2_predictor,
    processor,
    grounding_model,
    device,
):
    image = Image.open(img_path).convert("RGB")
    sam2_predictor.set_image(np.array(image))

    inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = grounding_model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.4,
        text_threshold=0.3,
        target_sizes=[image.size[::-1]],
    )

    if len(results) == 0 or len(results[0]["boxes"]) == 0:
        binary_mask = 255 * np.ones((image.height, image.width), dtype=np.uint8)
    else:
        input_boxes = results[0]["boxes"].cpu().numpy()
        masks, _, _ = sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )
        if masks.ndim == 4:
            masks = masks.squeeze(1)
        combined_mask = np.any(masks, axis=0)
        binary_mask = np.where(combined_mask, 0, 255).astype(np.uint8)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{img_path.stem}.png"
    cv2.imwrite(str(out_path), binary_mask)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, default="/home/shenzhen/Datasets/street_tryon/validation/image")
    parser.add_argument("--output-dir", type=str, default="/home/shenzhen/Datasets/street_tryon/validation/fg_masks")
    parser.add_argument("--grounding-model", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--text-prompt", default="person.")
    parser.add_argument("--sam2-checkpoint", default="./checkpoints/sam2.1_hiera_large.pt")
    parser.add_argument("--sam2-model-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--force-cpu", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu"

    sam2_model = build_sam2(args.sam2_model_config, args.sam2_checkpoint, device=device)
    sam2_predictor = SAM2ImagePredictor(sam2_model)

    processor = AutoProcessor.from_pretrained(args.grounding_model)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        args.grounding_model
    ).to(device)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    img_list = sorted([
        p for p in input_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".png", ".jpeg", ".webp")
    ])

    print(f"Found {len(img_list)} images")

    for img_path in tqdm(img_list, desc="Generating masks"):
        process_image(
            img_path,
            output_dir,
            args.text_prompt,
            sam2_predictor,
            processor,
            grounding_model,
            device,
        )


if __name__ == "__main__":
    main()