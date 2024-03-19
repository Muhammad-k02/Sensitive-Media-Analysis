import argparse
import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from scipy.ndimage import gaussian_filter

"""
The load_reference_video function takes in a video path and returns a list of frames.
    Args:
        video_path (str): The path to the reference video.

:param video_path: Specify the path of the video to be loaded
:return: A list of frames
"""
def load_reference_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames

"""
The get_video_frame_rate function takes in a video path and returns the frame rate of that video.

:param video_path: Specify the path to the video file
:return: The frame rate of the video
"""
def get_video_frame_rate(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return frame_rate


"""
The generate_heatmap_frames function takes in a video path, gaze positions, and an output directory.
It then creates a heatmap for each frame of the video using the gaze positions as input.
The heatmaps are saved to the output directory.

:param video_path: Load the reference video
:param gaze_positions: Specify the path to the gaze positions file
:param output_directory: Specify the directory where the heatmap frames will be saved
:return: An array of heatmaps for each frame in the video
"""
def generate_heatmap_frames(video_path, gaze_positions, output_directory):
    pd.options.display.float_format = '{:.3f}'.format
    surface_df = pd.read_csv(gaze_positions)
    offset = surface_df['world_index'].min()
    print("World index synchronized with Reference index")
    reference_frames = load_reference_video(video_path)
    print("Reference Frames Acquired")

    for frame_idx, reference_frame in enumerate(reference_frames):
        world_index = frame_idx + offset
        gaze_on_frame = surface_df[surface_df.world_index == world_index]
        print(f"Frame Index: {frame_idx}, World Index: {world_index}")

        grid = reference_frame.shape[0:2]
        heatmap_detail = 0.05

        gaze_on_frame_x = gaze_on_frame['norm_pos_x']
        gaze_on_frame_y = gaze_on_frame['norm_pos_y']
        gaze_on_frame_y = 1 - gaze_on_frame_y

        hist, x_edges, y_edges = np.histogram2d(
            gaze_on_frame_y,
            gaze_on_frame_x,
            range=[[0, 1.0], [0, 1.0]],
            normed=False,
            bins=grid
        )
        filter_h = int(heatmap_detail * grid[0]) // 2 * 2 + 1
        filter_w = int(heatmap_detail * grid[1]) // 2 * 2 + 1
        heatmap = gaussian_filter(hist, sigma=(filter_w, filter_h), order=0)
        print(f"Heatmap for Frame Index: {frame_idx} created")

        save_path = os.path.join(output_directory, f"frame_{frame_idx:04d}.png")
        plt.imshow(cv2.cvtColor(reference_frame, cv2.COLOR_BGR2RGB))
        plt.imshow(heatmap, cmap='jet', alpha=0.5)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()


"""
The main function is the entry point of the program.
It parses command line arguments, creates an output directory if it doesn't exist, gets the frame rate of the video from
ffprobe, generates heatmap frames and finally uses ffmpeg to create a video from those frames.
"""
def main():
    parser = argparse.ArgumentParser(description='Generate heatmap frames and create a video.')
    parser.add_argument('video_path', help='Path to the input video')
    parser.add_argument('gaze_positions', help='Path to the gaze positions CSV file')
    parser.add_argument('output_directory', help='Path to the output directory for frames and video')
    args = parser.parse_args()
    os.makedirs(args.output_directory, exist_ok=True)
    frame_rate = get_video_frame_rate(args.video_path)
    generate_heatmap_frames(args.video_path, args.gaze_positions, args.output_directory)
    ffmpeg_command = f"ffmpeg -framerate {frame_rate} -i {args.output_directory}/frame_%04d.png -c:v libx264 -r 30 -pix_fmt yuv420p {args.output_directory}/output_video.mp4"
    subprocess.run(ffmpeg_command, shell=True)

if __name__ == "__main__":
    main()





