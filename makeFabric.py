#!/usr/bin/env python3

import vtk
import numpy as np
import os
import sys
import glob
import re
import csv
from tqdm import tqdm

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
# Force-weighted fabric tensor (FAST)
# -------------------------------
def compute_fabric_force(polydata, name="force_normal"):

    arr = polydata.GetCellData().GetArray(name)

    if arr is None:
        raise ValueError(f"[ERROR] '{name}' not found in CellData")

    N = arr.GetNumberOfTuples()

    # ---- convert to numpy (FAST) ----
    forces = np.array([arr.GetTuple(i) for i in range(N)])

    mags = np.linalg.norm(forces, axis=1)

    valid = mags > 0
    if np.sum(valid) == 0:
        return np.zeros((3,3))

    n = forces[valid] / mags[valid][:, None]

    # ---- weighted fabric ----
    F = np.einsum('ni,nj,n->ij', n, n, mags[valid]) / np.sum(mags[valid])

    return F

# -------------------------------
# Main
# -------------------------------
def main():

    if len(sys.argv) < 2:
        print("Usage: python makeFabric.py <subdirectory>")
        sys.exit(1)

    subdir = sys.argv[1]

    if not os.path.isdir(subdir):
        print(f"[ERROR] Directory not found: {subdir}")
        sys.exit(1)

    # ---- output ----
    out_csv = f"fabric_{os.path.basename(subdir)}.csv"

    # ---- file list ----
    files = glob.glob(os.path.join(subdir, "*.vtp"))

    files = sorted(
        files,
        key=lambda x: extract_step(os.path.basename(x))
    )

    if len(files) == 0:
        print("[ERROR] No .vtp files found")
        sys.exit(1)

    print(f"\n[INFO] Subdir: {subdir}")
    print(f"[INFO] Files : {len(files)}")
    print(f"[INFO] Output: {out_csv}\n")

    # -------------------------------
    # CSV writing
    # -------------------------------
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

        # ---- progress bar ----
        for f in tqdm(files, desc="Fabric calc", unit="file"):

            fname = os.path.basename(f)
            step = extract_step(fname)

            # optional step log
            tqdm.write(f"[STEP] {step}")

            poly = read_vtp(f)

            # ---- core calculation ----
            F = compute_fabric_force(poly)

            vals, vecs = np.linalg.eigh(F)

            writer.writerow([
                os.path.basename(subdir),
                step,
                *F.flatten(),
                *vals
            ])

    print("\n[ DONE ] Fabric calculation completed.\n")

# -------------------------------
if __name__ == "__main__":
    main()