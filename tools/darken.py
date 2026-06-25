# Script to create artificial night like images from daytime images

# IMPORTS
import os
import argparse
import cv2
import numpy as np
from PIL import Image





# IMAGE PROCESSING FUNCTIONS

# Reduces brightness using gamma correction and scaling down brightness
def reduce_exposure(img, gamma, exposure_scale):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)]).astype("uint8")
    gamma_img = cv2.LUT(img, table)
    dark = (gamma_img.astype(np.float32) * exposure_scale).astype(np.uint8)
    return dark


# Shift towards blue colors, common at night
def color_shift_to_night(img, blue_strength, teal_strength):
    b, g, r = cv2.split(img.astype(np.float32) / 255.0)
    b = np.clip(b * (1.0 + blue_strength), 0, 1)
    g = np.clip(g * (1.0 + teal_strength), 0, 1)
    r = np.clip(r * (1.0 - (blue_strength + teal_strength) * 0.7), 0, 1)
    return (cv2.merge((b, g, r)) * 255).astype(np.uint8)


# Increases contrast using CLAHE to preserve detail
def local_contrast(img, clip_limit, tile_grid_size):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l2 = clahe.apply(l)
    lab2 = cv2.merge((l2,a,b))
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)


# Lifts shadows from pure black
def soften_shadows(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    v = hsv[:,:,2] / 255.0
    v = np.clip(v * (0.55 + 0.45 * v), 0, 1)
    hsv[:,:,2] = v * 255
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


# Add noise and slight grain, common in night images
def add_noise_and_grain(img, strength):
    noise = np.random.randn(*img.shape) * 255 * strength
    noisy = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy





# PROCESSING FUNCTIONS

# Process a single image
def process_image(img_path, out_path):

    # Load image
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(img_path)
    
    # Apply modifications
    img = reduce_exposure(img, gamma=1.9, exposure_scale=0.28)
    img = color_shift_to_night(img, blue_strength=0.22, teal_strength=0.10)
    img = local_contrast(img, clip_limit=2.0, tile_grid_size=(8,8))
    img = soften_shadows(img)
    img = add_noise_and_grain(img, strength=0.03)

    # Save image to output
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    pil.save(out_path, quality=95)


# Go through a folder and subfolder to find and process all images
def process_folder(input_dir, output_dir):

    # Image extensions
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']

    # Get paths
    input_dir = os.path.abspath(input_dir)
    output_dir = os.path.abspath(output_dir)

    # Abort if input folder doesn't exist
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input folder not found: {input_dir}")
    
    # Go over each element in the input folder
    for root, _, files in os.walk(input_dir):
        rel_root = os.path.relpath(root, input_dir)
        for fname in files:

            # Skip non images
            if not os.path.splitext(fname.lower())[1] in image_extensions:
                continue

            # Build paths
            in_path = os.path.join(root, fname)
            out_subdir = os.path.join(output_dir, rel_root) if rel_root != "." else output_dir
            out_fname = os.path.splitext(fname)[0] + "_dark" + os.path.splitext(fname)[1]
            out_path = os.path.join(out_subdir, out_fname)

            # Process each found image
            print(f"Processing: {in_path} -> {out_path}")
            try:
                process_image(in_path, out_path)
            except Exception as e:
                print(f"Failed {in_path}: {e}")





# MAIN
def main():

    # Parse the CLI arguments
    p = argparse.ArgumentParser(description="Batch convert daytime street photos to night-style images.")
    p.add_argument("input", help="Input file or folder")
    p.add_argument("output", help="Output file or folder")
    args = p.parse_args()

    if os.path.isfile(args.input):
        # Process a single file
        process_image(args.input, args.output)
        
    else:
        # Process a folder
        process_folder(args.input, args.output)

    return 0

if __name__ == "__main__":
    exit(main())
