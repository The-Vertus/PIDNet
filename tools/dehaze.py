# Preprocessing script to dehaze a foggy image

# IMPORTS
import cv2
import os
import numpy as np
import argparse
from pathlib import Path





# IMAGE PROCESSING FUNCTIONS

# Dehazes an image using the histogram method
def dehaze_histogram(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)
    
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


# Dehazes an image using the CLAHE approach
def dehaze_clahe(image, clip_limit):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

# Dehazes an image using the DCP approach, based on: He et al. "Single Image Haze Removal using Dark Channel Prior" (CVPR 2010)
def dehaze_dcp(image, patch_size, omega):

    # Convert to float for computation
    img_float = image.astype(np.float32) / 255.0
    
    # Compute dark channel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    
    # Min across spatial patch and color channels
    if len(image.shape) == 3:
        # Process each channel and take minimum
        b, g, r = cv2.split(img_float)
        dark_b = cv2.erode(b, kernel)
        dark_g = cv2.erode(g, kernel)
        dark_r = cv2.erode(r, kernel)
        dark_channel = np.minimum(dark_b, np.minimum(dark_g, dark_r))
    else:
        dark_channel = cv2.erode(img_float, kernel)
    
    # Estimate atmospheric light (A)
    # Use brightest pixel in dark channel as atmospheric light estimate
    # Find the brightest pixels in dark channel (most fogged areas)
    flat_dark = dark_channel.flatten()
    top_pixels_idx = np.argsort(-flat_dark)[:int(0.001 * len(flat_dark))]
    
    # Average the top 0.1% brightest pixels in dark channel
    A = np.zeros(3)
    for c_idx, channel in enumerate(cv2.split(img_float)):
        channel_flat = channel.flatten()
        A[c_idx] = np.mean(channel_flat[top_pixels_idx])
    A = np.maximum(A, 0.1)  # Ensure A is not too dark
    
    # Estimate transmission map
    # t(x) = 1 - omega * (dark_channel / A)
    transmission = np.zeros_like(dark_channel)
    for c in range(3):
        transmission = np.maximum(transmission, 
                                 1 - omega * (dark_channel / A[c]))
    
    transmission = np.clip(transmission, 0.1, 1.0)  # Prevent division by zero
    
    # For speed, simple bilateral filtering is used
    transmission_refined = cv2.bilateralFilter(
        (transmission * 255).astype(np.uint8), 
        d=9, 
        sigmaColor=75, 
        sigmaSpace=75
    ).astype(np.float32) / 255.0
    
    # Recover image
    output = np.zeros_like(img_float)
    for c in range(3):
        output[:, :, c] = (img_float[:, :, c] - A[c]) / transmission_refined + A[c]
    
    # Clip to valid range
    output = np.clip(output, 0, 1.0)
    
    # Convert back to uint8
    return (output * 255).astype(np.uint8)





# PROCESSING FUNCTIONS

# Process a single image
def process_image(img_path, out_path, method):

    # Load image
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(img_path)
    
    # Apply the processing
    if method == 'histogram':
        dehazed = dehaze_histogram(img)
    elif method == 'clahe':
        dehazed = dehaze_clahe(img, clip_limit=3.0)
    elif method == 'dcp':
        dehazed = dehaze_dcp(img, patch_size=30, omega=0.75)
    
    else:
        print(f"Unknown method: {method}")
        return

    # Save the image to output
    cv2.imwrite(out_path, dehazed)


# Go through a folder and subfolder to find and process all images
def process_folder(input_dir, output_dir, method):

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
            out_fname = os.path.splitext(fname)[0] + "_dehazed" + os.path.splitext(fname)[1]
            out_path = os.path.join(out_subdir, out_fname)

            # Process each found image
            print(f"Processing: {in_path} -> {out_path}")
            try:
                process_image(in_path, out_path, method)
            except Exception as e:
                print(f"Failed {in_path}: {e}")





# MAIN
def main():

    # Parse the CLI arguments
    p = argparse.ArgumentParser(description='Remduce fog in images to improve segmentation.')
    p.add_argument("input", help="Input file or folder")
    p.add_argument("output", help="Output file or folder")
    p.add_argument('-m', '--method', choices=['clahe', 'dcp', 'histogram'], default='dcp', help='Dehazing method (default: dcp)')
    args = p.parse_args()

    print(f'Using method: {args.method}')
    
    if os.path.isfile(args.input):
        # Process a single file
        process_image(args.input, args.output, args.method)

    else:
        # Process a folder
        process_folder(args.input, args.output, args.method)
    
    return 0

if __name__ == "__main__":
    exit(main())
