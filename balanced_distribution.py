import os
import random
from collections import defaultdict

# Configuration
num_people = 25
videos_per_person_per_class = 5

# Base directories where split videos are stored
split_base_dir = '/data/mkhan/twitter_selected/valid_videos/splitted_video'
label_0_dir = os.path.join(split_base_dir, "label_0")
label_1_dir = os.path.join(split_base_dir, "label_1")

# Allow user to choose a base directory for saving assignments
output_assignment_dir = "/data/mkhan/experimental_data/"
os.makedirs(output_assignment_dir, exist_ok=True)

def get_videos_from_subdirs(label_dir):
    """Extract videos from subdirectories, sampling one per subdirectory."""
    sampled_videos = []
    for subdir in os.listdir(label_dir):
        subdir_path = os.path.join(label_dir, subdir)
        if os.path.isdir(subdir_path):
            videos = [
                os.path.join(subdir_path, f) for f in os.listdir(subdir_path)
                if f.endswith(".mp4")
            ]
            if videos:
                sampled_videos.append(random.choice(videos))  # Randomly sample one video
    return sampled_videos

def get_remaining_videos(label_dir):
    """Get videos directly inside the base label directory (not in subdirectories)."""
    return [
        os.path.join(label_dir, f) for f in os.listdir(label_dir)
        if f.endswith(".mp4") and os.path.isfile(os.path.join(label_dir, f))
    ]

def assign_videos_to_subjects(label_0_videos, label_1_videos):
    """Assign 5 videos per class to each subject and store them in directories."""
    for i in range(num_people):
        subject_dir = os.path.join(output_assignment_dir, f"subject_{i + 1}")
        os.makedirs(subject_dir, exist_ok=True)

        # Create separate directories for each class
        label_0_subject_dir = os.path.join(subject_dir, "label_0")
        label_1_subject_dir = os.path.join(subject_dir, "label_1")
        os.makedirs(label_0_subject_dir, exist_ok=True)
        os.makedirs(label_1_subject_dir, exist_ok=True)

        # Select and move 5 videos for each label
        label_0_selection = random.sample(label_0_videos, videos_per_person_per_class)
        label_1_selection = random.sample(label_1_videos, videos_per_person_per_class)

        # Remove selected videos from the pool to avoid reuse
        label_0_videos = [v for v in label_0_videos if v not in label_0_selection]
        label_1_videos = [v for v in label_1_videos if v not in label_1_selection]

        # Move videos to the subject's directory
        move_videos(label_0_selection, label_0_subject_dir)
        move_videos(label_1_selection, label_1_subject_dir)

def move_videos(video_list, destination_dir):
    """Move or copy videos to the specified directory."""
    for video_path in video_list:
        file_name = os.path.basename(video_path)
        destination_path = os.path.join(destination_dir, file_name)
        os.rename(video_path, destination_path)  # Move the video
        print(f"Moved {video_path} -> {destination_path}")

# Extract videos from both classes
label_0_videos = get_videos_from_subdirs(label_0_dir) + get_remaining_videos(label_0_dir)
label_1_videos = get_videos_from_subdirs(label_1_dir) + get_remaining_videos(label_1_dir)

# Ensure we have enough videos for each subject
if len(label_0_videos) < num_people * videos_per_person_per_class and \
   len(label_1_videos) < num_people * videos_per_person_per_class:
    raise ValueError("Not enough videos available to assign to all subjects.")

# Assign videos to subjects and organize them in directories
assign_videos_to_subjects(label_0_videos, label_1_videos)

print(f"Video assignments completed. Check {output_assignment_dir} for results.")
