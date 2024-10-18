import argparse
import os
import numpy as np
import pandas as pd
import seaborn as sns

sns.set(context="notebook", style="whitegrid", font_scale=1.2)
import matplotlib.pyplot as plt
import matplotlib.colors as colors


def load_process(file_path):
    exported_gaze_file = file_path
    exported_gaze = pd.read_csv(exported_gaze_file).dropna()
    exported_gaze.info()
    print(exported_gaze.gaze_point_3d_z)
    negative_z_mask = exported_gaze.gaze_point_3d_z < 0
    negative_z_values = exported_gaze.loc[negative_z_mask, ["gaze_point_3d_z"]]
    exported_gaze.loc[negative_z_mask, ["gaze_point_3d_z"]] = negative_z_values * -1

    return exported_gaze


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


def sphere_pos_over_time(ts, data, unit="radians"):
    for key, values in data.items():
        sns.lineplot(x=ts, y=values, label=key)
        plt.xlabel("time [sec]")
        plt.ylabel(unit)
        plt.legend()


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


def gaze_velocity_calculation(exported_gaze, output_dir):
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

    # Save output files in the same directory as the gaze file
    plt.savefig(os.path.join(output_dir, 'gaze_velocity_scatter_plot.png'))

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

    # Save histogram in the same directory
    plt.savefig(os.path.join(output_dir, 'gaze_velocity_histogram.png'))


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


def main():
    parser = argparse.ArgumentParser(description='Generate Gaze Velocity')
    parser.add_argument('gaze_positions', help='Path to the gaze positions CSV file')
    args = parser.parse_args()

    # Get the output directory from the gaze file path
    output_dir = os.path.dirname(args.gaze_positions)

    gaze_data = load_process(args.gaze_positions)
    gaze_velocity_calculation(gaze_data, output_dir)
    r, theta, phi = cart_to_spherical(gaze_data)
    plot_on_sphere(r, theta, phi)


if __name__ == "__main__":
    main()
