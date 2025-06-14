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