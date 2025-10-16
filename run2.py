# import argparse
# import os
# import cv2
# import torch
# import numpy as np
# from PIL import Image
# from sam2.build_sam import build_sam2
# from sam2.sam2_image_predictor import SAM2ImagePredictor
# from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


# def process_image(image_path, text_prompt, sam2_predictor, processor, grounding_model, device, output_path):
#     image_name = os.path.basename(image_path)
#     base, ext = os.path.splitext(image_name)
#     mask_output_path = os.path.join(output_path, f"{base}_groundedsam2.png")

#     image = Image.open(image_path).convert("RGB")
#     sam2_predictor.set_image(np.array(image))

#     inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)
#     with torch.no_grad():
#         outputs = grounding_model(**inputs)
#     results = processor.post_process_grounded_object_detection(
#         outputs,
#         inputs.input_ids,
#         box_threshold=0.4,
#         text_threshold=0.3,
#         target_sizes=[image.size[::-1]]
#     )

#     input_boxes = results[0]["boxes"].cpu().numpy()

#     if input_boxes.shape[0] == 0:
#         binary_mask = 255 * np.ones((image.height, image.width), dtype=np.uint8)
#         cv2.imwrite(mask_output_path, binary_mask)
#         print(f"[{image_name}] No detection → saved all-white mask.")
#         return

#     masks, _, _ = sam2_predictor.predict(
#         point_coords=None,
#         point_labels=None,
#         box=input_boxes,
#         multimask_output=False,
#     )
#     if masks.ndim == 4:
#         masks = masks.squeeze(1)

#     combined_mask = np.any(masks, axis=0)
#     binary_mask = np.where(combined_mask, 0, 255).astype(np.uint8)
#     cv2.imwrite(mask_output_path, binary_mask)
#     print(f"[{image_name}] Saved mask → {mask_output_path}")


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--input-root-dir", type=str, required=True,
#                         help="Path containing train_A_raw, test_A_raw")
#     parser.add_argument("--text-prompt", default="person.", help="Prompt for Grounding-DINO")
#     parser.add_argument("--grounding-model", default="IDEA-Research/grounding-dino-tiny")
#     parser.add_argument("--sam2-checkpoint", default="./checkpoints/sam2.1_hiera_large.pt")
#     parser.add_argument("--sam2-model-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
#     parser.add_argument("--force-cpu", action="store_true")
#     args = parser.parse_args()

#     device = "cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu"

#     sam2_model = build_sam2(args.sam2_model_config, args.sam2_checkpoint, device=device)
#     sam2_predictor = SAM2ImagePredictor(sam2_model)

#     processor = AutoProcessor.from_pretrained(args.grounding_model)
#     grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(args.grounding_model).to(device)

#     torch.autocast(device_type=device, dtype=torch.bfloat16).__enter__()
#     if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
#         torch.backends.cuda.matmul.allow_tf32 = True
#         torch.backends.cudnn.allow_tf32 = True

#     subfolders = ["train_A_raw", "test_A_raw"] # NOTE: only used for source images now 
#     # subfolders = ["test_A_raw"] # NOTE: only used for source images now 

#     for folder_name in subfolders:
#         input_dir = os.path.join(args.input_root_dir, folder_name)
#         output_dir = os.path.join(input_dir)

#         if not os.path.exists(input_dir):
#             print(f"[SKIP] {input_dir} not found.")
#             continue

#         print(f"\n--- Processing {folder_name} ---")

#         for fname in sorted(os.listdir(input_dir)):

#             if not fname.lower().endswith(".png") or "_groundedsam2" in fname:
#                 continue  # skip non-png or already processed mask files

#             image_path = os.path.join(input_dir, fname)
#             process_image(image_path, args.text_prompt, sam2_predictor, processor, grounding_model, device, output_dir)


# if __name__ == "__main__":
#     main()