# Grayscale-video-colorization
This is an AI-powered video colorization pipeline that adds vibrant color to black-and-white video frames 🎞️✨

---

## 📽️ Project Pipeline (Coming Soon)

1. 🔽 Download video from YouTube  
2. 🎞️ Extract frames ✅ Implemented in [Frames.py](./Frames.py) 
3. 🌈 Colorize frames using deep learning  
4. 🎬 Convert to video  
5. 📊 Evaluate color quality

## 🎞️ Frame Extraction (Frames.py)

This script:
- Downloads a YouTube video using `yt-dlp`
- Starts from a custom time (like `01:30:00`)
- Extracts N seconds of frames at your chosen FPS (e.g., 30 fps)

Usage:
```bash
python Frames.py

---

## 📁 Project Structure

```bash
.
├── data/                   # Stores sample inputs and plots
│   ├── sample_frames/      # Sample input grayscale frames
│   └── plots/              # Metric visualizations
├── frames/                # Extracted grayscale frames from video
├── outputs/               # Colorized output frames
├── videos/                # Downloaded videos or converted outputs
├── results/               # Final colorized video and CSV evaluations
├── Frames.py              # YouTube downloader + frame extractor
├── frames_to_color.py     # Colorization using SIGGRAPH model
├── frames_to_video.py     # Frame-to-video stitcher
├── evaluation_and_plot.py # LPIPS, PSNR, Delta E, and colorfulness analysis
├── get-pip.py             # Optional pip installer
├── requirements.txt       # All required packages
└── README.md              # You’re looking at it 👀

## 🎬 Demo Snapshot

Here’s a peek at the colorization in action 👇

▶️ [Watch output video](./results/output_video_colorfulNature.mp4)

> From grayscale to vibrant — one frame at a time 🌈


🔧 Project Overview
This project brings life and color to black-and-white videos!
Built using PyTorch and computer vision tools, the pipeline:

Downloads grayscale video clips 🎥

Extracts frames at custom times and frame rates 🕒

Applies deep learning to colorize each frame 🌈

Evaluates quality using LPIPS, ΔE, and PSNR metrics 📊

It’s an end-to-end automated solution for turning the past vivid again ✨

## ⚙️ How to Run the Pipeline

Make sure dependencies are installed:

```bash
pip install -r requirements.txt

1️⃣ Extract Frames:
python Frames.py

What it does:
Downloads video from YouTube
Starts at your chosen time
Saves extracted frames in /frames or /data/sample_frames

2️⃣ Colorize the Frames
python frames_to_color.py -i frames/ -o outputs/ --use_gpu

- Requires PyTorch & pretrained model
- Use --use_gpu for faster processing

3️⃣ Convert Frames to Video
python frames_to_video.py

What it does:
Stitches colorized PNGs into output_video.mp4

4️⃣ Evaluate Quality
python evaluation_and_plot.py

What it does:
Calculates LPIPS, ΔE, UV_PSNR, and colorfulness
Saves metric plots to data/plots/




