import os
import cv2
import subprocess

def download_youtube_video(url, save_path="video10.mp4"):
    # Downloading YouTube video using yt-dlp
    save_path = "video10.mp4"
    command = ["python", "-m", "yt_dlp", "-f", "136", "-o", save_path, url]


    
    subprocess.run(command, check=True)
    print(f"Video downloaded as {save_path}")
    return save_path

def convert_time_to_seconds(time_str):
            parts = list(map(int, time_str.split(":")))
            if len(parts) == 3:
                hours, minutes, seconds = parts
            elif len(parts) == 2:
                hours = 0
                minutes, seconds = parts
            else:
                raise ValueError("Time format must be MM:SS or HH:MM:SS")
    
            return hours * 3600 + minutes * 60 + seconds

def extract_frames(video_path, output_folder="frames10", start_time="1:00", duration=10, fps=30):
    # Extract frames from a given start time for a specified duration
    
    start_time_sec = convert_time_to_seconds(start_time)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
# Open the video with OpenCV’s capture object.
    cap = cv2.VideoCapture(video_path)
    # Jump the playback cursor to the chosen start time (in milliseconds).
    cap.set(cv2.CAP_PROP_POS_MSEC, start_time_sec * 1000)
# Calculate how many frames to save. Example: 10 s × 30 fps = 300 frames.
    total_frames = int(duration * fps)
    frame_count = 0

    while cap.isOpened():
        # ret is False when there are no more frames.
        ret, frame = cap.read()
        if not ret or frame_count >= total_frames:
            break
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1

    cap.release()
    print(f"Extracted {frame_count} frames from {start_time} to {start_time_sec + duration}s in {output_folder}")

# Provide the YouTube video URL
youtube_url = "https://www.youtube.com/watch?v=Tvjx50NGKeo"


# Step 1: Download the YouTube video
video_file = download_youtube_video(youtube_url)
# video_file = "C:/Users/aditi/OneDrive/Desktop/CAPSTONE/colorization-master/Videos/video5.mp4"

# Step 2: Extract frames
extract_frames(video_file, output_folder="video10", start_time="01:30:30", duration=20, fps=30)
