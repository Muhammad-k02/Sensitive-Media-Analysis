import os
import random
import subprocess
import pandas as pd

# Configurations
num_people = 25
videos_per_person_per_class = 5
files_per_label = 5
label_0 = 0
label_1 = 1
output_base_dir = "/path/to/save/split_videos"  # Specify the save directory

required_videos_per_label = files_per_label * num_people


def run_ffmpeg_command(cmd):
    """Run FFmpeg command and handle errors."""
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e}")


def process_video(input_path, save_dir):
    """Process the video using FFmpeg."""
    print(f"Processing: {input_path}")

    # Get video duration
    cmd_duration = [
        'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1', input_path
    ]
    duration = float(subprocess.check_output(cmd_duration).strip())
    print(f"Loaded: {input_path}, Duration: {duration:.2f} seconds")

    base_name = os.path.splitext(os.path.basename(input_path))[0]
    video_dir = os.path.join(save_dir, base_name)
    os.makedirs(video_dir, exist_ok=True)

    if duration < 60:  # Save short videos directly
        output_path = os.path.join(video_dir, f"{base_name}.mp4")
        cmd_save = ['ffmpeg', '-i', input_path, '-c:v', 'libx264', output_path]
        run_ffmpeg_command(cmd_save)
        print(f"Saved short video: {output_path}")
        return

    # Split the video into 5 parts or with a leftover
    total_duration = min(duration, 300)  # Cap at 5 minutes
    segment_duration = total_duration / 5
    segments = []
    start = 0

    for i in range(5):
        end = min(start + segment_duration, total_duration)
        segment_path = os.path.join(video_dir, f"part_{i + 1}.mp4")
        cmd_split = [
            'ffmpeg', '-i', input_path, '-ss', str(start), '-to', str(end),
            '-c:v', 'libx264', segment_path
        ]
        run_ffmpeg_command(cmd_split)
        segments.append(segment_path)
        print(f"Segment {i + 1}: {end - start:.2f} seconds")
        start = end
        if start >= total_duration:
            break

    if duration > total_duration:
        leftover_path = os.path.join(video_dir, f"leftover.mp4")
        cmd_leftover = [
            'ffmpeg', '-i', input_path, '-ss', str(total_duration),
            '-c:v', 'libx264', leftover_path
        ]
        run_ffmpeg_command(cmd_leftover)
        segments.append(leftover_path)
        print(f"Leftover Segment: {duration - total_duration:.2f} seconds")

    print(f"Processed {input_path} into {len(segments)} segments.")


def filter_videos_by_duration(video_paths, min_duration=10):
    """Filter videos by minimum duration using FFmpeg."""
    valid_videos = []
    for path in video_paths:
        try:
            cmd_duration = [
                'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1', path
            ]
            duration = float(subprocess.check_output(cmd_duration).strip())
            if duration >= min_duration:
                valid_videos.append(path)
            else:
                print(f"Skipped {path}: Duration {duration:.2f} seconds (too short)")
        except Exception as e:
            print(f"Error loading {path}: {e}")
    return valid_videos


# Load CSV files
df_nonviolence = pd.read_csv(
    '/data/datasets/Violence_Dataset/VideoMae_Data/combo5/data_preservation/working_combioned.csv', header=None)
df_violence = pd.read_csv('/data/mkhan/twitter_selected/valid_videos/codes/twitter_violence.csv', header=None)
df3 = pd.read_csv('/data/datasets/Violence_Dataset/RWF2000/RWF-2000/combined_total.csv', header=None)

# Extract video paths by label
label_0_files = df_nonviolence[df_nonviolence[1] == 0][0].tolist()
label_1_files = df_violence[df_violence[1] == 1][0].tolist()
label_1_files_extra = df3[df3[1] == 1][0].tolist()

print(f"Initial label 0 videos: {len(label_0_files)}")
print(f"Initial label 1 videos: {len(label_1_files)}")
print(f"Available extra label 1 videos: {len(label_1_files_extra)}")

label_0_files = filter_videos_by_duration(label_0_files)
label_1_files = filter_videos_by_duration(label_1_files)
label_1_files_extra = filter_videos_by_duration(label_1_files_extra)


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


def process_videos(video_files, label):
    """Process and split videos into labeled directories."""
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
