#!/usr/bin/env python3

import vtk
import numpy as np
import os
import sys
import glob
import re
import csv
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# -------------------------------
# FONT (Cambria - 강제 적용)
# -------------------------------
font_path = "/Library/Fonts/cambria.ttc"
font_prop = fm.FontProperties(fname=font_path)

def apply_font(ax):
    for item in (
        [ax.title, ax.xaxis.label, ax.yaxis.label] +
        ax.get_xticklabels() + ax.get_yticklabels()
    ):
        item.set_fontproperties(font_prop)
        item.set_fontsize(14)

# -------------------------------
# VTP reader
# -------------------------------
def read_vtp(filename):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()

# -------------------------------
# Extract step number
# -------------------------------
def extract_step(fname):
    m = re.search(r'(\d+)', fname)
    return int(m.group(1)) if m else 0

# -------------------------------
# Fabric tensor (force-weighted)
# -------------------------------
def compute_fabric_force(polydata, name="force_normal"):

    arr = polydata.GetCellData().GetArray(name)

    if arr is None:
        raise ValueError(f"[ERROR] '{name}' not found")

    N = arr.GetNumberOfTuples()

    forces = np.array([arr.GetTuple(i) for i in range(N)])

    mags = np.linalg.norm(forces, axis=1)

    valid = mags > 0
    if np.sum(valid) == 0:
        return np.zeros((3,3))

    n = forces[valid] / mags[valid][:, None]

    F = np.einsum('ni,nj,n->ij', n, n, mags[valid]) / np.sum(mags[valid])

    return F

# -------------------------------
# MAIN
# -------------------------------
def main():

    if len(sys.argv) < 2:
        print("Usage: python makeFabricPlot.py <subdirectory>")
        sys.exit(1)

    subdir = sys.argv[1]

    if not os.path.isdir(subdir):
        print(f"[ERROR] Directory not found: {subdir}")
        sys.exit(1)

    out_csv = f"fabric_{os.path.basename(subdir)}.csv"

    files = sorted(
        glob.glob(os.path.join(subdir, "*.vtp")),
        key=lambda x: extract_step(os.path.basename(x))
    )

    if len(files) == 0:
        print("[ERROR] No .vtp files found")
        sys.exit(1)

    print(f"\n[INFO] Subdir: {subdir}")
    print(f"[INFO] Files : {len(files)}")
    print(f"[INFO] Output: {out_csv}\n")

    # -------------------------------
    # CSV + data collection
    # -------------------------------
    steps, e1_list, e3_list = [], [], []

    with open(out_csv, "w", newline="") as fcsv:

        writer = csv.writer(fcsv)

        writer.writerow([
            "subdir",
            "step",
            "F11","F12","F13",
            "F21","F22","F23",
            "F31","F32","F33",
            "e1","e2","e3"
        ])

        for f in tqdm(files, desc="Fabric calc", unit="file"):

            fname = os.path.basename(f)
            step = extract_step(fname)

            tqdm.write(f"[STEP] {step}")

            poly = read_vtp(f)

            F = compute_fabric_force(poly)
            vals, vecs = np.linalg.eigh(F)

            writer.writerow([
                os.path.basename(subdir),
                step,
                *F.flatten(),
                *vals
            ])

            # store for plot
            steps.append(step)
            e1_list.append(vals[0])
            e3_list.append(vals[2])

    print("\n[INFO] CSV saved.\n")

    # -------------------------------
    # Plot anisotropy
    # -------------------------------
    steps = np.array(steps)
    e1 = np.array(e1_list)
    e3 = np.array(e3_list)

    anis = e3 - e1

    fig, ax = plt.subplots()

    ax.plot(steps, anis, marker='o', linewidth=1.5)

    ax.set_xlabel("Step")
    ax.set_ylabel("Anisotropy (e3 - e1)")
    ax.set_title(f"Fabric Anisotropy ({os.path.basename(subdir)})")

    ax.grid()

    apply_font(ax)

    plt.tight_layout()
    plt.savefig(f"{os.path.basename(subdir)}_anisotropy.png", dpi=300)
    plt.show()

    print("[ DONE ] Fabric + anisotropy plot completed.\n")

# -------------------------------
if __name__ == "__main__":
    main()
