import argparse
import os
import matplotlib.pyplot as plt
# colorizers is the helper module from the Zhang/Daniel Liu repository that wraps their ECCV’16 and SIGGRAPH’17 networks and utility functions.
from colorizers import *
from PIL import Image
import torch


def init_colorizer(use_gpu):
    model = siggraph17(pretrained=True).eval()
    if use_gpu:
        model.cuda()
    return model


# Function to process each frame with SIGGRAPH17
# Function to process each frame with SIGGRAPH17
def process_frame(img_path,colorizer, use_gpu, save_prefix_bw, save_prefix_color):
    # Load the SIGGRAPH17 colorizer model, eval() puts the model in inference/ output generation mode instead of training
    # colorizer_siggraph17 = siggraph17(pretrained=True).eval()

    # if use_gpu:
    #     colorizer_siggraph17.cuda()                                     //now loading it in init_colorizer

    # Load image and preprocess
    # load_img opens the image as an RGB NumPy array.
    img = load_img(img_path)
    # preprocess_img converts to LAB, extracts the L channel twice:
    # tens_l_orig = original resolution L (to preserve detail),
    # tens_l_rs = resized 256 × 256 L (network’s native input size).
    # Both are returned as PyTorch tensors with shape [1, 1, H, W].
    (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256, 256))

    if use_gpu:
        tens_l_rs = tens_l_rs.cuda()

    # Create outputs
    # img_bw is just the grayscale reference: L plus zeroed a & b.
    # torch.cat((0*L, 0*L)) creates empty chroma channels.
    img_bw = postprocess_tens(tens_l_orig, torch.cat((0 * tens_l_orig, 0 * tens_l_orig), dim=1))

    # colorizer_siggraph17(tens_l_rs) → network predicts [1, 2, 256, 256] tensor of ab.
    # .cpu() brings it back to CPU memory.
    # postprocess_tens upsamples ab to match tens_l_orig size, combines with L, converts back to RGB, then outputs a NumPy array in [0,1] range.

    out_img_siggraph17 = postprocess_tens(tens_l_orig, colorizer(tens_l_rs).cpu())

    # Save both results
    plt.imsave(f'{save_prefix_bw}.png', img_bw, cmap='gray')
    plt.imsave(f'{save_prefix_color}.png', out_img_siggraph17)


# Main function to loop through all frames in the folder
# Main function to loop through all frames in the folder
def process_all_frames(frames_folder, output_folder, use_gpu):
    # Create output subfolders
    bw_folder = os.path.join(output_folder, "bw")
    color_folder = os.path.join(output_folder, "color")
    os.makedirs(bw_folder, exist_ok=True)
    os.makedirs(color_folder, exist_ok=True)

    colorizer = init_colorizer(use_gpu)

    # List all files in the frames folder
    for frame_name in os.listdir(frames_folder):
        frame_path = os.path.join(frames_folder, frame_name)

        if frame_name.endswith(('.jpg', '.jpeg', '.png')):
            base_name = os.path.splitext(frame_name)[0]
            save_prefix_bw = os.path.join(bw_folder, base_name)
            save_prefix_color = os.path.join(color_folder, base_name)

            process_frame(frame_path, colorizer, use_gpu, save_prefix_bw, save_prefix_color)
            print(f"Processed {frame_name}")


# Command-line arguments for the program
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--frames_folder', type=str, required=True, help='Folder containing the input frames')
parser.add_argument('-o', '--output_folder', type=str, required=True, help='Folder to save the colorized frames')
parser.add_argument('--use_gpu', action='store_true', help='Whether to use GPU')
opt = parser.parse_args()

# Run the processing
process_all_frames(opt.frames_folder, opt.output_folder, opt.use_gpu)
