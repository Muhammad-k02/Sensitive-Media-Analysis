import pathlib

import numpy as np
import pandas as pd
import seaborn as sns

sns.set(context="notebook", style="whitegrid", font_scale=1.2)

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import argparse

"""
The pupil_calculation function takes two arguments:
    - pupil_positions_file: a csv file containing the pupil positions for each frame of the video.
    - fixations_file: a csv file containing the fixation points and their durations.

:param pupil_positions_file: Specify the file path of the pupil positions data
:param fixations_file: Specify the file that contains the fixations
:return: A list of lists
"""
def pupil_calculation(pupil_positions_file, fixations_file):
    pupil_positions_file = pupil_positions_file
    fixations_file = fixations_file

    print("Necessary files:")
    print(pupil_positions_file)
    print(fixations_file)

    pupil_positions = pd.read_csv(pupil_positions_file)
    fixations = pd.read_csv(fixations_file)

    pupil_positions = pupil_positions[pupil_positions.method != "2d c++"]

    pupil_positions = pupil_positions[["pupil_timestamp", "diameter_3d", "confidence"]]
    fixations = fixations[["id", "start_timestamp", "duration"]]

    pupil_positions.info()
    print()
    fixations.info()

    MIN_CONFIDENCE = 0.8

    results = []

    for _, fixations in fixations.groupby("id"):
        first_fixation = fixations.iloc[0]
        print(first_fixation)

        fixation_id = first_fixation.id
        fixation_start = first_fixation.start_timestamp
        fixation_end = fixation_start + first_fixation.duration / 1000

        mask_after_start = fixation_start <= pupil_positions.pupil_timestamp
        mask_before_end = pupil_positions.pupil_timestamp <= fixation_end
        mask_high_confidence = pupil_positions.confidence >= MIN_CONFIDENCE

        pupil_positions_during_fixation = pupil_positions[mask_after_start & mask_before_end & mask_high_confidence]
        diameter_3d_during_fixation = pupil_positions_during_fixation.diameter_3d

        results.append([fixation_id, diameter_3d_during_fixation.mean()])

    return results


"""
The main function of this module is to generate a scatter plot of the mean pupil diameter 3d by fixation.
The function takes three arguments:
    1) The path to the pupil_positions_file CSV file.
    2) The path to the fixations_file CSV file.
    3) A string that will be used as a suffix for naming and saving output files.

:return: The mean pupil diameter for each fixation
"""


def main():
    parser = argparse.ArgumentParser(description='Generate Pupil Diameter Diagram')
    parser.add_argument('pupil_positions_file', help='Path to the pupil_positions_file CSV ')
    parser.add_argument('fixations_file', help='Path to the fixations_file CSV file')
    parser.add_argument('output_file_suffix', help='output file directory name')

    args = parser.parse_args()

    mean_diameter_3d_by_fixation = pd.DataFrame(pupil_calculation(args.pupil_positions_file, args.fixations_file),
                                                columns=["id", "mean_pupil_diameter_3d"])

    plt.scatter(mean_diameter_3d_by_fixation.id, mean_diameter_3d_by_fixation.mean_pupil_diameter_3d)
    max_diameter = mean_diameter_3d_by_fixation['mean_pupil_diameter_3d'].max()
    min_diameter = mean_diameter_3d_by_fixation['mean_pupil_diameter_3d'].min()

    plt.xlabel("fixation_id")
    plt.ylabel("mean diameter_3d [mm]")
    plt.ylim(min_diameter - 0.5, max_diameter + 0.5)
    plt.savefig('pupil_diameter ' + args.output_file_suffix + '.png')
    plt.show()


if __name__ == "__main__":
    main()
