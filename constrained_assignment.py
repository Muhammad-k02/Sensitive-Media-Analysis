import itertools
from collections import defaultdict
import pandas as pd
import shutil
import os
import random
from moviepy.video.io.VideoFileClip import VideoFileClip

# Configuration
num_people = 25
videos_per_class = 25
videos_per_person_per_class = 5
files_per_label = 5
label_0 = 0
label_1 = 1


# Helper functions
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def copy_files(file_list, target_dir):
    for file_path in file_list:
        shutil.copy(file_path, target_dir)


# Load CSV files without headers
df_nonviolence = pd.read_csv(
    '/data/datasets/Violence_Dataset/VideoMae_Data/combo5/data_preservation/working_combioned.csv',
    header=None
)

df_violence = pd.read_csv(
    '/data/mkhan/twitter_selected/valid_videos/codes/twitter_violence.csv',
    header=None
)

df3 = pd.read_csv(
    '/data/datasets/Violence_Dataset/RWF2000/RWF-2000/combined_total.csv',
    header=None
)

# Extract file paths based on labels
label_0_files = df_nonviolence[df_nonviolence[1] == 0][0].tolist()
label_1_files = df_violence[df_violence[1] == 1][0].tolist()
label_1_files_extra = df3[df3[1] == 1][0].tolist()

print(len(label_0_files), len(label_1_files))

# Calculate required videos
required_videos_per_label = files_per_label * num_people


# Check if enough videos are available
def check_video_availability(files, label):
    missing = max(0, required_videos_per_label - len(files))
    if missing > 0:
        print(f"Label {label}: Missing {missing} videos.")
    else:
        print(f"Label {label}: All required videos are available.")
    return missing


missing_label_0 = check_video_availability(label_0_files, 0)
missing_label_1 = check_video_availability(label_1_files, 1)

# Handle missing videos for the violence class
if missing_label_1 > 0:
    # Calculate how many extra videos are needed
    extra_needed = missing_label_1
    # Take the required amount of videos from label_1_files_extra
    if extra_needed <= len(label_1_files_extra):
        sampled_extra_videos = random.sample(label_1_files_extra, extra_needed)
        label_1_files.extend(sampled_extra_videos)
        print(f"Added {extra_needed} videos from label_1_files_extra to label_1_files.")
    else:
        print(f"Not enough extra videos available. Need {extra_needed}, but only have {len(label_1_files_extra)}.")

# Optional: Display final counts after adding extra videos
print(f"Final count of label 1 videos: {len(label_1_files)}")

# Stop execution if there aren't enough videos
if len(label_0_files) < required_videos_per_label or len(label_1_files) < required_videos_per_label:
    raise ValueError("Not enough videos to meet the requirements.")


# Splits the input video into 60-second segments and returns them
def split_video(input_path):
    """Splits the input video into 60-second segments."""
    clip = VideoFileClip(input_path)
    print(f"Loaded: {input_path}, Duration: {clip.duration}s")

    segments = []
    full_minutes = int(clip.duration // 60)

    # Create segments of 60 seconds each
    for i in range(full_minutes):
        start = i * 60
        end = start + 60
        sub_clip = clip.subclip(start, end)
        segment_path = f"{input_path}_part_{i + 1}.mp4"  # Save segment with a unique name
        sub_clip.write_videofile(segment_path)  # Save segment to file
        segments.append(segment_path)  # Add segment path to the list

    # Handle the remaining portion, if any
    remainder = clip.duration % 60
    if remainder > 0:
        sub_clip = clip.subclip(full_minutes * 60, clip.duration)
        segment_path = f"{input_path}_part_{full_minutes + 1}.mp4"  # Save last segment
        sub_clip.write_videofile(segment_path)
        segments.append(segment_path)  # Add last segment path to the list

    clip.close()
    print(f"Video {input_path} split into {len(segments)} segments.")
    return segments


# Function to ensure all videos are one minute or less
# Function to ensure all videos are one minute or less
def process_videos(files):
    processed_videos = []

    for file in files:
        clip = VideoFileClip(file)
        if clip.duration > 60:
            print(f"Video {file} is longer than 60 seconds, splitting...")
            segments = split_video(file)
            # Randomly select one of the segments to replace the original video
            selected_segment = random.choice(segments)
            processed_videos.append(selected_segment)  # Add only the selected segment
        else:
            processed_videos.append(file)  # Keep the original if it's <= 60 seconds
        clip.close()

    return processed_videos

# Process videos for both labels
label_0_files, label_0_split_videos = process_videos(label_0_files)
label_1_files, label_1_split_videos = process_videos(label_1_files)

# Combine original videos and split videos
label_0_files.extend(label_0_split_videos)
label_1_files.extend(label_1_split_videos)


# Randomly select from split videos to fill the class if needed
def fill_class_from_split(class_files, split_videos):
    while len(class_files) < required_videos_per_label:
        if split_videos:
            selected_video = random.choice(split_videos)
            class_files.append(selected_video)
            split_videos.remove(selected_video)
        else:
            break


fill_class_from_split(label_0_files, label_0_split_videos)
fill_class_from_split(label_1_files, label_1_split_videos)


# Create balanced assignment function
def create_balanced_assignment(videos, num_people, videos_per_person):
    all_combinations = list(itertools.combinations(videos, videos_per_person))
    video_count = defaultdict(int)
    assignments = []

    while len(assignments) < num_people:
        comb = random.choice(all_combinations)
        if all(video_count[video] < 10 for video in comb):
            assignments.append(comb)
            for video in comb:
                video_count[video] += 1

    return assignments


# Create dummy video sets for balanced assignments
class_a_videos = [f"A_video_{i + 1}" for i in range(videos_per_class)]
class_b_videos = [f"B_video_{i + 1}" for i in range(videos_per_class)]

assignments_a = create_balanced_assignment(class_a_videos, num_people, videos_per_person_per_class)
assignments_b = create_balanced_assignment(class_b_videos, num_people, videos_per_person_per_class)

# Map assigned videos to each person
people_videos = {
    f"Person_{i + 1}": {
        "Class_A": list(assignments_a[i]),
        "Class_B": list(assignments_b[i])
    }
    for i in range(num_people)
}

# Copy videos to individual directories
for person_id in range(num_people):
    person_key = f"Person_{person_id + 1}"
    target_dir = f'/data/mkhan/experimental_data/Subject_{person_id + 1}'
    create_dir(target_dir)

    # Get the videos for this
    label_0_subset = label_0_files[:files_per_label]
    label_1_subset = label_1_files[:files_per_label]

    # Copy the videos
    copy_files(label_0_subset, target_dir)
    copy_files(label_1_subset, target_dir)

    # Remove copied files from the list
    label_0_files = label_0_files[files_per_label:]
    label_1_files = label_1_files[files_per_label:]

# Print assignments and video watch count
for person_id, videos in people_videos.items():
    print(f"{person_id}:")
    print(f"  Class A Videos: {videos['Class_A']}")
    print(f"  Class B Videos: {videos['Class_B']}")
    print()

# Count how many times each video was assigned
video_watch_count = defaultdict(int)
for videos in people_videos.values():
    for video in videos['Class_A'] + videos['Class_B']:
        video_watch_count[video] += 1

# Print video watch counts
for video, count in video_watch_count.items():
    print(f"Video {video} was watched {count} times.")

print("Files copied successfully.")
