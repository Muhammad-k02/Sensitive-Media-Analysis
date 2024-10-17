import os
import random
from moviepy.editor import VideoFileClip
import pandas as pd

# Configurations
num_people = 25
videos_per_person_per_class = 5
files_per_label = 5
label_0 = 0
label_1 = 1
output_base_dir = "/path/to/save/split_videos"  # Specify where to save split videos

required_videos_per_label = files_per_label * num_people

# Function to split videos or save short ones directly
def process_video(input_path, save_dir):
    """Process the video: Split if longer than a minute, save directly if shorter."""
    clip = VideoFileClip(input_path)
    print(f"Loaded: {input_path}, Duration: {clip.duration:.2f} seconds")

    base_name = os.path.splitext(os.path.basename(input_path))[0]
    video_dir = os.path.join(save_dir, base_name)
    os.makedirs(video_dir, exist_ok=True)

    if clip.duration < 60:  # Save short videos directly
        output_path = os.path.join(video_dir, f"{base_name}.mp4")
        clip.write_videofile(output_path, codec="libx264")
        print(f"Saved short video: {output_path}")
        clip.close()
        return

    # Split video into 5 parts or handle leftover duration
    total_duration = min(clip.duration, 300)  # Limit to 5 minutes
    segment_duration = total_duration / 5
    segments = []
    start = 0

    for i in range(5):
        end = min(start + segment_duration, total_duration)
        sub_clip = clip.subclip(start, end)
        segment_path = os.path.join(video_dir, f"part_{i + 1}.mp4")
        sub_clip.write_videofile(segment_path, codec="libx264")
        segments.append(segment_path)
        print(f"Segment {i + 1}: {end - start:.2f} seconds")
        start = end
        if start >= total_duration:
            break

    if clip.duration > total_duration:
        leftover_clip = clip.subclip(total_duration, clip.duration)
        leftover_path = os.path.join(video_dir, f"leftover.mp4")
        leftover_clip.write_videofile(leftover_path, codec="libx264")
        segments.append(leftover_path)
        print(f"Leftover Segment: {clip.duration - total_duration:.2f} seconds")

    clip.close()
    print(f"Processed {input_path} into {len(segments)} segments.")

# Load CSV files
df_nonviolence = pd.read_csv('/data/datasets/Violence_Dataset/VideoMae_Data/combo5/data_preservation/working_combioned.csv', header=None)
df_violence = pd.read_csv('/data/mkhan/twitter_selected/valid_videos/codes/twitter_violence.csv', header=None)
df3 = pd.read_csv('/data/datasets/Violence_Dataset/RWF2000/RWF-2000/combined_total.csv', header=None)

# Extract video paths by label
label_0_files = df_nonviolence[df_nonviolence[1] == 0][0].tolist()
label_1_files = df_violence[df_violence[1] == 1][0].tolist()
label_1_files_extra = df3[df3[1] == 1][0].tolist()

print(f"Initial label 0 videos: {len(label_0_files)}")
print(f"Initial label 1 videos: {len(label_1_files)}")
print(f"Available extra label 1 videos: {len(label_1_files_extra)}")

def check_video_availability(files, label):
    """Check if enough videos are available and return the number of missing videos."""
    missing = max(0, required_videos_per_label - len(files))
    if missing > 0:
        print(f"Label {label}: Missing {missing} videos.")
    else:
        print(f"Label {label}: All required videos available.")
    return missing

missing_label_0 = check_video_availability(label_0_files, 0)
missing_label_1 = check_video_availability(label_1_files, 1)

# Add extra videos for label 1 if needed
if missing_label_1 > 0:
    extra_needed = missing_label_1
    if extra_needed <= len(label_1_files_extra):
        sampled_extra_videos = random.sample(label_1_files_extra, extra_needed)
        label_1_files.extend(sampled_extra_videos)
        print(f"Added {extra_needed} extra videos to label 1.")
    else:
        print(f"Not enough extra videos available. Needed: {extra_needed}, Found: {len(label_1_files_extra)}")

print(f"Final label 1 videos: {len(label_1_files)}")

# Ensure we have the required number of videos
if len(label_0_files) < required_videos_per_label or len(label_1_files) < required_videos_per_label:
    raise ValueError("Not enough videos to meet the requirements.")

def process_videos(video_files, label):
    """Process and split videos, saving them under the appropriate label directory."""
    label_dir = os.path.join(output_base_dir, f"label_{label}")
    os.makedirs(label_dir, exist_ok=True)

    for video_path in video_files:
        try:
            process_video(video_path, label_dir)
        except Exception as e:
            print(f"Error processing {video_path}: {e}")

print("Processing label 0 videos...")
process_videos(label_0_files[:required_videos_per_label], label_0)

print("Processing label 1 videos...")
process_videos(label_1_files[:required_videos_per_label], label_1)
