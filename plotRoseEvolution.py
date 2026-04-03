#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import vtk
import sys
import glob
import os
import re
from tqdm import tqdm
import imageio

# -------------------------------
# FONT (Cambria)
# -------------------------------
font_path = "/Library/Fonts/cambria.ttc"
font_prop = fm.FontProperties(fname=font_path)

def apply_font(ax):
    for item in [ax.title]:
        item.set_fontproperties(font_prop)
        item.set_fontsize(12)

    for label in ax.get_xticklabels():
        label.set_fontproperties(font_prop)
        label.set_fontsize(10)

# -------------------------------
# Read VTP
# -------------------------------
def read_vtp(filename):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()

# -------------------------------
# Extract step
# -------------------------------
def extract_step(fname):
    m = re.search(r'(\d+)', fname)
    return int(m.group(1)) if m else 0

# -------------------------------
# Extract normals
# -------------------------------
def get_force_normals(polydata, name="force_normal"):

    arr = polydata.GetCellData().GetArray(name)

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
# Principal direction
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

    ax.plot([angle, angle],
            [0, ax.get_ylim()[1]],
            color='red', linewidth=2)

# -------------------------------
# Rose
# -------------------------------
def compute_rose(normals, weights, plane):

    if plane == "XY":
        a = np.arctan2(normals[:,1], normals[:,0])
    elif plane == "XZ":
        a = np.arctan2(normals[:,2], normals[:,0])
    elif plane == "YZ":
        a = np.arctan2(normals[:,2], normals[:,1])

    a = np.mod(a, 2*np.pi)

    hist, bin_edges = np.histogram(a, bins=36, weights=weights)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    return centers, hist

# -------------------------------
# MAIN
# -------------------------------
def main():

    if len(sys.argv) < 2:
        print("Usage: python plotRoseEvolution.py <subdirectory>")
        sys.exit(1)

    subdir = sys.argv[1]

    files = sorted(
        glob.glob(os.path.join(subdir, "*.vtp")),
        key=lambda x: extract_step(os.path.basename(x))
    )

    os.makedirs("frames", exist_ok=True)

    images = []

    for i, f in enumerate(tqdm(files, desc="Processing")):

        step = extract_step(os.path.basename(f))

        poly = read_vtp(f)
        normals, weights = get_force_normals(poly)
        F = compute_fabric(normals, weights)

        fig, axes = plt.subplots(
            1, 3,
            subplot_kw={'projection': 'polar'},
            figsize=(12,4)
        )

        planes = ["XY", "XZ", "YZ"]

        for ax, plane in zip(axes, planes):

            c, h = compute_rose(normals, weights, plane)

            ax.bar(c, h, width=2*np.pi/36)

            overlay_principal(ax, F, plane)

            ax.set_title(f"{plane} | Step {step}",
                         fontproperties=font_prop)

            ax.set_theta_zero_location("E")
            ax.set_theta_direction(-1)

            apply_font(ax)

        plt.tight_layout()

        fname = f"frames/frame_{i:04d}.png"
        plt.savefig(fname, dpi=200)
        plt.close()

        images.append(imageio.imread(fname))

    # -------------------------------
    # GIF 생성
    # -------------------------------
    imageio.mimsave("rose_evolution.gif", images, fps=5)

    print("\n[ DONE ] Animation saved: rose_evolution.gif\n")

# -------------------------------
if __name__ == "__main__":
    main()
