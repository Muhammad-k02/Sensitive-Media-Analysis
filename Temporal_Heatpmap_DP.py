import argparse
import os
from tqdm import tqdm
import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

"""
The load_reference_video function takes in a video path and returns a list of frames.
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
The get_video_frame_rate function returns the frame rate of the video.
"""
def get_video_frame_rate(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return frame_rate

"""
Generate heatmap video from gaze data.
"""
def generate_heatmap_video(video_path, gaze_positions, output_path, frame_rate):
    surface_df = pd.read_csv(gaze_positions)

    # Check if required columns exist
    if not {'world_index', 'norm_pos_x', 'norm_pos_y'}.issubset(surface_df.columns):
        print("Error: CSV file does not contain required columns.")
        return

    reference_frames = load_reference_video(video_path)
    if not reference_frames:
        print("Error: No frames loaded from video.")
        return

    height, width, _ = reference_frames[0].shape
    grid = (height // 2, width // 2)

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, frame_rate, (width, height))

    offset = surface_df['world_index'].min()

    for frame_idx, reference_frame in tqdm(enumerate(reference_frames), total=len(reference_frames)):
        world_index = frame_idx + offset

        if world_index not in surface_df['world_index'].values:
            continue

        gaze_on_frame = surface_df[surface_df.world_index == world_index]
        gaze_x = gaze_on_frame['norm_pos_x'].values
        gaze_y = 1 - gaze_on_frame['norm_pos_y'].values  # Adjust for OpenCV's coordinate system

        # Generate heatmap
        hist, _, _ = np.histogram2d(gaze_y * height, gaze_x * width, bins=grid, range=[[0, height], [0, width]], density=True)
        heatmap = gaussian_filter(hist, sigma=(5, 5))

        # Normalize and color the heatmap
        heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap_resized = cv2.resize(heatmap_colored, (width, height))

        # Blend the heatmap with the original frame
        overlay = cv2.addWeighted(reference_frame, 0.7, heatmap_resized, 0.3, 0)
        video_writer.write(overlay)

    video_writer.release()
    print(f"Heatmap video successfully written to {output_path}")
"""
Generate fixation video by plotting fixation points.
"""


def generate_fixation_video(video_path, fixation_data, output_path, frame_rate):
    fixation_df = pd.read_csv(fixation_data)

    # Ensure required columns are present
    required_cols = {'start_frame_index', 'end_frame_index', 'norm_pos_x', 'norm_pos_y', 'id'}
    if not required_cols.issubset(fixation_df.columns):
        print("Error: Fixation CSV file does not contain required columns.")
        return

    reference_frames = load_reference_video(video_path)
    if not reference_frames:
        print("Error: No frames loaded from video.")
        return

    height, width, _ = reference_frames[0].shape

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, frame_rate, (width, height))

    offset = fixation_df['start_frame_index'].min()

    for frame_idx, reference_frame in tqdm(enumerate(reference_frames), total=len(reference_frames)):
        adjusted_frame_index = frame_idx + offset

        # Filter fixation data for the current frame
        current_fixations = fixation_df[
            (fixation_df['start_frame_index'] <= adjusted_frame_index) &
            (fixation_df['end_frame_index'] >= adjusted_frame_index)
        ]

        # Skip if no fixations for the current frame
        if current_fixations.empty:
            video_writer.write(reference_frame)
            continue

        # Plot fixations directly on the frame
        for _, fixation in current_fixations.iterrows():
            gaze_x = int(fixation['norm_pos_x'] * width)
            gaze_y = int(fixation['norm_pos_y'] * height)  # Adjusted for OpenCV's top-left origin

            # Draw a red circle for fixation
            cv2.circle(reference_frame, (gaze_x, gaze_y), 10, (0, 0, 255), -1)

            # Draw fixation ID as text
            cv2.putText(
                reference_frame, str(fixation['id']),
                (gaze_x + 5, gaze_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
            )

        # Write the modified frame to the video
        video_writer.write(reference_frame)

    video_writer.release()
    print(f"Fixation video successfully written to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Generate heatmap and fixation videos.')
    parser.add_argument('video_path', help='Path to the input video')
    parser.add_argument('gaze_positions', help='Path to the gaze positions CSV file')
    parser.add_argument('fixation_data', help='Path to the fixation data CSV file')
    parser.add_argument('heatmap_video', help='Path to the output heatmap video')
    parser.add_argument('fixation_video', help='Path to the output fixation video')
    args = parser.parse_args()

    frame_rate = get_video_frame_rate(args.video_path)
    generate_heatmap_video(args.video_path, args.gaze_positions, args.heatmap_video, frame_rate)
    generate_fixation_video(args.video_path, args.fixation_data, args.fixation_video, frame_rate)

if __name__ == "__main__":
    main()
