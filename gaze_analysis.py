import argparse
import os
import numpy as np
import pandas as pd
import seaborn as sns

sns.set(context="notebook", style="whitegrid", font_scale=1.2)
import matplotlib.pyplot as plt
import matplotlib.colors as colors

"""
The load_process function takes in a file path and returns the processed dataframe.
The function reads in the csv file, drops all NaN values, and then converts negative z-values to positive.
:param file_path: Specify the path to the exported gaze data file
:return: A pandas dataframe
"""


def load_process(file_path):
    exported_gaze_file = file_path
    exported_gaze = pd.read_csv(exported_gaze_file).dropna()
    exported_gaze.info()
    print(exported_gaze.gaze_point_3d_z)
    negative_z_mask = exported_gaze.gaze_point_3d_z < 0
    negative_z_values = exported_gaze.loc[negative_z_mask, ["gaze_point_3d_z"]]
    exported_gaze.loc[negative_z_mask, ["gaze_point_3d_z"]] = negative_z_values * -1

    return exported_gaze


"""
The cart_to_spherical function converts 3D Cartesian coordinates to spherical coordinates.
:param data: Pass in the dataframe
:param apply_rad2deg: Convert the angles from radians to degrees
:return: The radius, theta and phi values
"""


def cart_to_spherical(data, apply_rad2deg=False):
    x = data.gaze_point_3d_x
    y = data.gaze_point_3d_y
    z = data.gaze_point_3d_z
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = np.arccos(y / r)
    phi = np.arctan2(z, x)

    if apply_rad2deg:
        theta = np.rad2deg(theta)
        phi = np.rad2deg(phi)

    return r, theta, phi


"""
The sphere_pos_over_time function plots the position of a sphere over time.
:param ts: Plot the x-axis of the graph
:param data: Pass the data to the function
:param unit: Set the y-axis label
:return: The position of the sphere over time
"""


def sphere_pos_over_time(ts, data, unit="radians"):
    for key, values in data.items():
        sns.lineplot(x=ts, y=values, label=key)
        plt.xlabel("time [sec]")
        plt.ylabel(unit)
        plt.legend()


"""
The sphere_pos function takes in three arrays of equal length, r, theta and phi.
It then plots a scatter plot of theta vs phi with each point colored by its distance from
the origin (r). The colorbar is logarithmic to better show points that are close to the origin.
The unit parameter allows you to specify whether or not you want your angles in radians or degrees.

:param r: Set the color of each point
:param theta: Set the x-axis of the plot, and phi is used to set the y-axis
:param phi: Set the angle of rotation around the z-axis
:param unit: Set the units of the x and y axis labels
:return: A scatter plot of theta and phi with color
"""


def sphere_pos(r, theta, phi, unit="radians"):
    print(r.min(), r.max())
    norm = colors.LogNorm(vmin=r.min(), vmax=r.max())
    points = plt.scatter(
        theta,
        phi,
        c=r,
        alpha=0.5,
        cmap="cubehelix",
        norm=norm,
    )
    cbar = plt.colorbar(points)
    cbar.ax.set_ylabel("distance [mm]", rotation=270)
    cbar.ax.get_yaxis().labelpad = 15
    plt.xlabel(f"theta [{unit[:3]}]")
    plt.ylabel(f"phi [{unit[:3]}]")


"""
The gaze_velocity_calculation function calculates the velocity of the gaze over time.
It takes in a dataframe with gaze coordinates and outputs two plots: one showing the 
gaze velocity over time, and another showing a histogram of all velocities. The function 
also prints out some statistics about these velocities.

:param exported_gaze: Pass the dataframe of gaze points to the function
:return: The gaze velocity over time
"""


def gaze_velocity_calculation(exported_gaze):
    print('Constructing Gaze Velocity Diagram')
    r, theta, phi = cart_to_spherical(exported_gaze, apply_rad2deg=True)
    plt.figure(figsize=(16, 4))

    plt.subplot(1, 2, 1)
    sphere_pos_over_time(
        exported_gaze.gaze_timestamp,
        data={"theta": theta, "phi": phi},
        unit="degrees"
    )

    plt.subplot(1, 2, 2)
    sphere_pos(r, theta, phi, unit="degrees")
    plt.savefig('gaze_velocity_scatter_plot.png')
    plt.show()

    squared_theta_diff = np.diff(theta) ** 2
    squared_phi_diff = np.diff(phi) ** 2
    deg_diff = np.sqrt(squared_theta_diff + squared_phi_diff)
    ts_diff = np.diff(exported_gaze.gaze_timestamp)
    deg_per_sec = deg_diff / ts_diff

    time = exported_gaze.gaze_timestamp[:-1] - exported_gaze.gaze_timestamp.iloc[0]

    plt.figure(figsize=(16, 4))

    plt.subplot(1, 2, 1)
    sphere_pos_over_time(time, {"gaze velocity": deg_per_sec}, unit="deg/sec")
    plt.title("Gaze velocity over time")

    plt.subplot(1, 2, 2)
    plt.hist(deg_per_sec, bins=np.logspace(-1, np.log10(500), 50))
    plt.title("Gaze velocity histogram")
    plt.xlabel("Gaze velocity [deg/sec]")
    plt.savefig('gaze_velocity_histogram.png')
    plt.show()



"""
The plot_on_sphere function takes in three arrays of equal length, r, theta and phi.
The function then plots these points on a 3D sphere using matplotlib's 3D plotting capabilities.
The user can specify whether they want to use radians or degrees for the angles 
:param r: Set the radius of the sphere
:param theta: Set the angle of elevation
:param phi: Set the angle of rotation around the z axis
:param unit: Specify the units of theta and phi
:return: A 3d scatter plot of points on a sphere
"""
def plot_on_sphere(r, theta, phi, unit="radians"):

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    ax.scatter(x, y, z, c=r, cmap='viridis')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Points on a Sphere')

    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Generate Gaze Velocity')
    parser.add_argument('gaze_positions', help='Path to the gaze positions CSV file')
    args = parser.parse_args()
    gaze_velocity_calculation(load_process(args.gaze_positions))
    r, theta, phi = cart_to_spherical(load_process(args.gaze_positions))
    plot_on_sphere(r, theta, phi)


if __name__ == "__main__":
    main()
