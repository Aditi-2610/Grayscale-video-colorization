import os
import cv2
import numpy as np
import torch
import lpips
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage import color
from colormath.color_diff import delta_e_cie2000
import matplotlib.pyplot as plt
import csv

# --- Setup LPIPS ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lpips_model = lpips.LPIPS(net='alex').to(device)

# --- Helper Functions ---

def load_image_rgb(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found at: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def image_to_tensor(img):
    img = img.astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(device)

def calculate_lpips(img1, img2):
    t1, t2 = image_to_tensor(img1), image_to_tensor(img2)
    with torch.no_grad():
        score = lpips_model(t1, t2)
    return score.item()

def calculate_uv_psnr(img1, img2):
    # Convert RGB to YUV
    yuv1 = cv2.cvtColor(img1, cv2.COLOR_RGB2YUV)
    yuv2 = cv2.cvtColor(img2, cv2.COLOR_RGB2YUV)
    # Extract U and V channels
    u1, v1 = yuv1[..., 1], yuv1[..., 2]
    u2, v2 = yuv2[..., 1], yuv2[..., 2]
    # Calculate PSNR for U and V and average
    psnr_u = compare_psnr(u1, u2, data_range=255)
    psnr_v = compare_psnr(v1, v2, data_range=255)
    return (psnr_u + psnr_v) / 2.0

def calculate_colorfulness(img):
    R, G, B = img[..., 0], img[..., 1], img[..., 2]
    rg = np.abs(R - G)
    yb = np.abs(0.5 * (R + G) - B)
    std_rg, std_yb = np.std(rg), np.std(yb)
    mean_rg, mean_yb = np.mean(rg), np.mean(yb)
    return np.sqrt(std_rg**2 + std_yb**2) + 0.3 * np.sqrt(mean_rg**2 + mean_yb**2)

def calculate_delta_e(img1, img2):
    lab1 = color.rgb2lab(img1.astype(np.float32) / 255.0)
    lab2 = color.rgb2lab(img2.astype(np.float32) / 255.0)
    delta_e = color.deltaE_ciede2000(lab1, lab2)
    return np.mean(delta_e)

def plot_metric(values, title, ylabel, color='skyblue', save_dir='plots', ideal_value=None):
    import os
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(10, 4))
    plt.plot(values, marker='o', linestyle='-', color=color, label=title)

    avg_val = np.mean(values)
    min_val = np.min(values)
    max_val = np.max(values)

    plt.axhline(y=avg_val, color='red', linestyle='--', label=f'Avg: {avg_val:.2f}')
    plt.axhline(y=min_val, color='green', linestyle='-.', label=f'Min: {min_val:.2f}')
    plt.axhline(y=max_val, color='green', linestyle=':', label=f'Max: {max_val:.2f}')

    if ideal_value is not None:
        plt.axhline(y=ideal_value, color='green', linestyle='-', linewidth=2, label=f'Ideal: {ideal_value}')

    plt.title(f"{title} per Image")
    plt.xlabel("Image Index")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    filename = f"{title.lower().replace(' ', '_')}.png"
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot: {save_path}")


# --- Main Evaluation Function ---

def evaluate_and_plot(ground_truth_dir, predicted_dir):
    lpips_scores = []
    psnr_scores = []
    delta_e_scores = []
    colorfulness_scores = []

    for gt_filename in os.listdir(ground_truth_dir):
        if gt_filename.endswith('.jpg'):
            gt_path = os.path.join(ground_truth_dir, gt_filename)
            pred_filename = gt_filename.replace('.jpg', '.png')
            pred_path = os.path.join(predicted_dir, pred_filename).replace("\\", "/")

            img_gt = load_image_rgb(gt_path)
            img_pred = load_image_rgb(pred_path)
            if img_gt is None or img_pred is None:
                continue

            # LPIPS
            lpips_score = calculate_lpips(img_gt, img_pred)
            lpips_scores.append(lpips_score)

            # UV_PSNR
            psnr_score = calculate_uv_psnr(img_gt, img_pred)
            psnr_scores.append(psnr_score)

            # Delta E
            delta_e_score = calculate_delta_e(img_gt, img_pred)
            delta_e_scores.append(delta_e_score)


            # Colorfulness
            colorfulness_score = calculate_colorfulness(img_pred)
            colorfulness_scores.append(colorfulness_score)

            print(f'{gt_filename} | LPIPS: {lpips_score:.4f} | PSNR: {psnr_score:.2f} | ΔE: {delta_e_score:.2f} | Colorfulness: {colorfulness_score:.2f}')

    print("\nAverages across all images:")
    print(f"Avg LPIPS: {np.mean(lpips_scores):.4f}")
    print(f"Avg PSNR: {np.mean(psnr_scores):.2f}")
    print(f"Avg Delta E: {np.mean(delta_e_scores):.2f}")
    print(f"Avg Colorfulness: {np.mean(colorfulness_scores):.2f}")

        # Save results to CSV
    csv_file_path = os.path.join(predicted_dir, "evaluation_results.csv")
    with open(csv_file_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Image", "LPIPS", "UV_PSNR", "Delta_E", "Colorfulness"])

        for i, gt_filename in enumerate(os.listdir(ground_truth_dir)):
            if gt_filename.endswith('.jpg'):
                pred_filename = gt_filename.replace('.jpg', '.png')
                writer.writerow([
                    pred_filename,
                    f"{lpips_scores[i]:.4f}",
                    f"{psnr_scores[i]:.2f}",
                    f"{delta_e_scores[i]:.2f}",
                    f"{colorfulness_scores[i]:.2f}"
                ])

    print(f"\nSaved evaluation results to: {csv_file_path}")


    # Plot and save graphs
    plot_metric(lpips_scores, "LPIPS Score", "LPIPS", ideal_value=0.0)
    plot_metric(psnr_scores, "UV_PSNR Score", "PSNR (UV channels)", ideal_value=30.0)
    plot_metric(delta_e_scores, "Delta E (CIEDE2000)", "ΔE", ideal_value=0.0)
    plot_metric(colorfulness_scores, "Colorfulness", "Colorfulness Score")  # No ideal value for this one



# --- Run ---

ground_truth_dir = r"C:/Users/aditi/OneDrive/Desktop/CAPSTONE/colorization-master/video_frames_bnw1"
predicted_dir = r"C:/Users/aditi/OneDrive/Desktop/CAPSTONE/colorization-master/frames_colored_bnw1/color"


evaluate_and_plot(ground_truth_dir, predicted_dir)