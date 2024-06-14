import cv2
import pandas as pd
import os
import csv

def extract_i_frames(video_path, output_dir, label, csv_writer):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    i_frame_count = 0

    video_base_name = os.path.splitext(os.path.basename(video_path))[0]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_type = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if frame_type % 250 == 0:
            output_path = os.path.join(output_dir, f"{video_base_name}_frame_{i_frame_count:04d}.jpg")
            cv2.imwrite(output_path, frame)
            csv_writer.writerow([output_path, label])
            i_frame_count += 1

        frame_count += 1

    cap.release()
    print(f"Extracted {i_frame_count} I-frames from {frame_count} frames in video {video_path}")

def process_videos(csv_path, output_dir, output_csv_path):
    df = pd.read_csv(csv_path, header=None)
    video_paths = df[0].tolist()
    labels = df[1].tolist()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['image_path', 'label'])

        for video_path, label in zip(video_paths, labels):
            extract_i_frames(video_path, output_dir, label, csv_writer)

if __name__ == "__main__":
    csv_path = 'path_to_your_video_dataset.csv'
    output_dir = 'output_frames_directory'
    output_csv_path = 'output_labels.csv'

    process_videos(csv_path, output_dir, output_csv_path)
