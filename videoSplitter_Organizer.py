import os
import random
import pandas as pd
import subprocess
import shutil

# Configurations
num_people = 25
videos_per_person_per_class = 5
files_per_label = 5
label_0 = 0
label_1 = 1
output_base_dir = "/data/mkhan/experimental_dataV2"  # Specify where to save split videos

required_videos_per_label = files_per_label * num_people

# Function to split videos or save short ones directly using ffmpeg
def process_video(input_path, save_dir):
    """Process the video: Split if longer than a minute, save directly if shorter."""
    print(f"Processing: {input_path}")

    base_name = os.path.splitext(os.path.basename(input_path))[0]
    video_dir = os.path.join(save_dir, base_name)
    os.makedirs(video_dir, exist_ok=True)

    # Get the duration of the video using ffmpeg
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", input_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    duration = float(result.stdout)

    print(f"Loaded: {input_path}, Duration: {duration:.2f} seconds")

    if duration < 60:  # Save short videos directly
        output_path = os.path.join(video_dir, f"{base_name}.mp4")
        subprocess.run(["ffmpeg", "-i", input_path, "-c:v", "libx264", output_path])
        print(f"Saved short video: {output_path}")
        return

    # Split video into 5 parts or handle leftover duration
    total_duration = min(duration, 300)  # Limit to 5 minutes
    segment_duration = total_duration / 5
    segments = []
    start = 0

    for i in range(5):
        end = min(start + segment_duration, total_duration)
        segment_path = os.path.join(video_dir, f"part_{i + 1}.mp4")
        subprocess.run(["ffmpeg", "-i", input_path, "-ss", str(start), "-to", str(end), "-c:v", "libx264", segment_path])
        segments.append(segment_path)
        print(f"Segment {i + 1}: {end - start:.2f} seconds")
        start = end
        if start >= total_duration:
            break

    if duration > total_duration:
        leftover_path = os.path.join(video_dir, f"leftover.mp4")
        subprocess.run(["ffmpeg", "-i", input_path, "-ss", str(total_duration), "-c:v", "libx264", leftover_path])
        segments.append(leftover_path)
        print(f"Leftover Segment: {duration - total_duration:.2f} seconds")

    print(f"Processed {input_path} into {len(segments)} segments.")

def filter_videos_by_duration(video_paths, min_duration=10):
    """Filter out videos with a duration less than the minimum duration."""
    valid_videos = []
    for path in video_paths:
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            duration = float(result.stdout)
            if duration >= min_duration:
                valid_videos.append(path)
            else:
                print(f"Skipped {path}: Duration {duration:.2f} seconds (too short)")
        except Exception as e:
            print(f"Error loading {path}: {e}")
    return valid_videos

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

# Filter videos for label 0 only
label_0_files = filter_videos_by_duration(label_0_files)

# Helper to remove duplicate paths
def remove_duplicates(file_list):
    """Remove duplicates from the list while preserving order."""
    return list(dict.fromkeys(file_list))

# Remove duplicates in label 1 videos
label_1_files = remove_duplicates(label_1_files)
label_1_files_extra = remove_duplicates(label_1_files_extra)

def check_video_availability(files, label):
    """Check if enough videos are available and return the number of missing videos."""
    missing = max(0, required_videos_per_label - len(files))
    if missing > 0:
        print(f"Label {label}: Missing {missing} videos.")
    else:
        print(f"Label {label}: All required videos available.")
    return missing

missing_label_1 = check_video_availability(label_1_files, 1)

# Add extra videos for label 1 if needed and save immediately
if missing_label_1 > 0:
    extra_needed = missing_label_1
    available_extras = [v for v in label_1_files_extra if v not in label_1_files]
    if extra_needed <= len(available_extras):
        sampled_extra_videos = random.sample(available_extras, extra_needed)
        label_1_files.extend(sampled_extra_videos)
        print(f"Added {extra_needed} extra videos to label 1.")
    else:
        print(f"Not enough extra videos available. Needed: {extra_needed}, Found: {len(available_extras)}")

print(f"Final label 1 videos: {len(label_1_files)}")

# Save label 1 videos immediately
def save_videos_directly(video_files, label):
    """Save videos directly to the label directory without processing."""
    label_dir = os.path.join(output_base_dir, f"label_{label}")
    os.makedirs(label_dir, exist_ok=True)

    for video_path in video_files:
        try:
            dest_path = os.path.join(label_dir, os.path.basename(video_path))
            shutil.copy(video_path, dest_path)
            print(f"Saved: {dest_path}")
        except Exception as e:
            print(f"Error saving {video_path}: {e}")

print("Saving label 1 videos directly...")
save_videos_directly(label_1_files[:required_videos_per_label], 1)

# Check availability of label 0 videos and process them
missing_label_0 = check_video_availability(label_0_files, 0)

if missing_label_0 > 0:
    raise ValueError("Not enough videos for label 0.")

print("Processing label 0 videos...")
for video_path in label_0_files[:required_videos_per_label]:
    process_video(video_path, os.path.join(output_base_dir, f"label_0"))