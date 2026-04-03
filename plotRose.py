#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import vtk
import sys

# -------------------------------
# FONT (Cambria 강제 적용)
# -------------------------------
font_path = "/Library/Fonts/cambria.ttc"
font_prop = fm.FontProperties(fname=font_path)

def apply_font(ax):
    for item in [ax.title]:
        item.set_fontproperties(font_prop)
        item.set_fontsize(14)

    for label in ax.get_xticklabels():
        label.set_fontproperties(font_prop)
        label.set_fontsize(12)

# -------------------------------
# Read VTP
# -------------------------------
def read_vtp(filename):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()

# -------------------------------
# Extract force normals
# -------------------------------
def get_force_normals(polydata, name="force_normal"):

    arr = polydata.GetCellData().GetArray(name)

    if arr is None:
        raise ValueError(f"[ERROR] '{name}' not found")

    N = arr.GetNumberOfTuples()

    forces = np.array([arr.GetTuple(i) for i in range(N)])

    mags = np.linalg.norm(forces, axis=1)
    valid = mags > 0

    normals = forces[valid] / mags[valid][:, None]
    weights = mags[valid]

    return normals, weights

# -------------------------------
# Fabric tensor
# -------------------------------
def compute_fabric(normals, weights):
    return np.einsum('ni,nj,n->ij', normals, normals, weights) / np.sum(weights)

# -------------------------------
# Principal direction overlay
# -------------------------------
def overlay_principal(ax, F, plane):

    vals, vecs = np.linalg.eigh(F)
    v = vecs[:, np.argmax(vals)]

    if plane == "XY":
        angle = np.arctan2(v[1], v[0])
    elif plane == "XZ":
        angle = np.arctan2(v[2], v[0])
    elif plane == "YZ":
        angle = np.arctan2(v[2], v[1])

    ax.plot(
        [angle, angle],
        [0, ax.get_ylim()[1]],
        color='red',
        linewidth=2.5
    )

# -------------------------------
# Rose histogram
# -------------------------------
def compute_rose(normals, weights, plane):

    if plane == "XY":
        angles = np.arctan2(normals[:,1], normals[:,0])
    elif plane == "XZ":
        angles = np.arctan2(normals[:,2], normals[:,0])
    elif plane == "YZ":
        angles = np.arctan2(normals[:,2], normals[:,1])

    angles = np.mod(angles, 2*np.pi)

    hist, bin_edges = np.histogram(
        angles,
        bins=36,
        weights=weights
    )

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    return bin_centers, hist

# -------------------------------
# MAIN
# -------------------------------
def main():

    if len(sys.argv) < 2:
        print("Usage: python plotRose.py <vtp_file>")
        sys.exit(1)

    vtp_file = sys.argv[1]

    print(f"[INFO] VTP: {vtp_file}")

    poly = read_vtp(vtp_file)
    normals, weights = get_force_normals(poly)

    F = compute_fabric(normals, weights)

    # -------------------------------
    # 3-plane plot
    # -------------------------------
    fig, axes = plt.subplots(
        1, 3,
        subplot_kw={'projection': 'polar'},
        figsize=(15, 5)
    )

    planes = ["XY", "XZ", "YZ"]

    for ax, plane in zip(axes, planes):

        bin_centers, hist = compute_rose(normals, weights, plane)

        ax.bar(bin_centers, hist, width=2*np.pi/36)

        overlay_principal(ax, F, plane)

        ax.set_title(
            f"{plane} Plane",
            fontproperties=font_prop
        )

        ax.set_theta_zero_location("E")
        ax.set_theta_direction(-1)

        apply_font(ax)

    plt.tight_layout()
    plt.savefig("rose_3planes.png", dpi=300)
    plt.show()

    print("\n[ DONE ] Rose diagram generated.\n")

# -------------------------------
if __name__ == "__main__":
    main()
