import subprocess
import os

def create_video_from_frames(input_folder, output_video):
    # Ensure the input folder is correct and update pattern for your frame names
    input_pattern = os.path.join(input_folder, "frame_%04d_siggraph17.png")

    # Build the ffmpeg command
    ffmpeg_path = r"C:/Users/aditi/OneDrive/Desktop/CAPSTONE/colorization-master/ffmpeg/bin/ffmpeg.exe"  # Use the correct path to ffmpeg.exe
    command = [
        ffmpeg_path,
        "-framerate", "24",  # Set frame rate (adjust as needed)
        "-i", input_pattern,  # Input frame pattern
        "-c:v", "libx264",    # Video codec
        "-pix_fmt", "yuv420p", # Pixel format for compatibility
        output_video          # Output video file
    ]

    # Run the command
    subprocess.run(command, check=True)

# Example usage
create_video_from_frames(r'C:/Users/aditi/OneDrive/Desktop/CAPSTONE/colorization-master/frames_colored', 'output_video.mp4')
