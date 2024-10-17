import itertools
import random
from collections import defaultdict

# Configuration
num_people = 25
videos_per_person_per_class = 5

# Load video files from CSVs (replace with actual paths)
label_0_files = pd.read_csv(
    '/data/datasets/Violence_Dataset/VideoMae_Data/combo5/data_preservation/working_combioned.csv',
    header=None
)[0].tolist()

label_1_files = pd.read_csv(
    '/data/mkhan/twitter_selected/valid_videos/codes/twitter_violence.csv',
    header=None
)[0].tolist()

# Ensure we have enough videos for balanced assignment
required_videos_per_label = num_people * videos_per_person_per_class

if len(label_0_files) < required_videos_per_label or len(label_1_files) < required_videos_per_label:
    raise ValueError("Not enough videos available for balanced assignment.")

# Randomly sample the required number of videos for each class
label_0_files = random.sample(label_0_files, required_videos_per_label)
label_1_files = random.sample(label_1_files, required_videos_per_label)

# Create balanced assignments for each class
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

# Generate assignments for both labels
assignments_label_0 = create_balanced_assignment(label_0_files, num_people, videos_per_person_per_class)
assignments_label_1 = create_balanced_assignment(label_1_files, num_people, videos_per_person_per_class)

# Store assigned videos for each person
people_videos = {
    f"Person_{i + 1}": {
        "Class_0": list(assignments_label_0[i]),
        "Class_1": list(assignments_label_1[i])
    }
    for i in range(num_people)
}

# Print out the assignments for each person
for person, videos in people_videos.items():
    print(f"{person}:")
    print(f"  Class 0 Videos: {videos['Class_0']}")
    print(f"  Class 1 Videos: {videos['Class_1']}")
    print()

# Count how many times each video was assigned
video_watch_count = defaultdict(int)
for videos in people_videos.values():
    for video in videos['Class_0'] + videos['Class_1']:
        video_watch_count[video] += 1

# Print video watch counts
for video, count in video_watch_count.items():
    print(f"Video {video} was watched {count} times.")
