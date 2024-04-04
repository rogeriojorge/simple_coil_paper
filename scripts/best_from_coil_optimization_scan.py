#!/usr/bin/env python3
import os
import json
import glob
import shutil
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from simsopt import load
import matplotlib.pyplot as plt
from paretoset import paretoset
import matplotlib.colors as colors
from simsopt.geo import SurfaceRZFourier
this_path = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument("--type", type=int, default=1)
args = parser.parse_args()

print_coil_currents = False

if args.type == 1: QA_or_QH = 'simple'
elif args.type == 2: QA_or_QH = 'QA'
elif args.type == 3: QA_or_QH = 'QH'
elif args.type == 4: QA_or_QH = 'QI'
elif args.type == 5: QA_or_QH = 'QI_nfp2'
else: raise ValueError('Invalid type')

nphi = 60
ntheta = 60
use_nfp3 = True

print(f'This is {QA_or_QH}')
results_path = os.path.join(os.path.dirname(__file__), 'results_'+QA_or_QH,'scan')
Path(results_path).mkdir(parents=True, exist_ok=True)
os.chdir(results_path)

# Initialize an empty DataFrame
df = pd.DataFrame()

results = glob.glob("*/results.json")
for results_file in results:
    with open(results_file, "r") as f:
        data = json.load(f)

    # Wrap lists in another list
    for key, value in data.items():
        if isinstance(value, list):
            data[key] = [value]

    df = pd.concat([df, pd.DataFrame(data)], ignore_index=True)
print(f'Number of runs of {QA_or_QH}: {len(df)}')
df = df[df["max_max_curvature"] < 50]

#########################################################
# Here you can define criteria to filter out the most interesting runs.
#########################################################

succeeded = df["linking_number"] < 0.1
succeeded = np.logical_and(succeeded, df["coil_coil_distance"] > 0.05)
succeeded = np.logical_and(succeeded, df["Jf"] < 1e-2)
succeeded = np.logical_and(succeeded, df["max_max_curvature"] < 25)
# succeeded = np.logical_and(succeeded, df["coil_surface_distance1"] > 0.049)
# succeeded = np.logical_and(succeeded, df["coil_surface_distance2"] > 0.049)
# succeeded = np.logical_and(succeeded, df["length"] < 17)
# succeeded = np.logical_and(succeeded, df["BdotN"] < 1.9e-2)

#########################################################
# End of filtering criteria
#########################################################

df_filtered = df[succeeded]
print(f'Number of succeeded runs of {QA_or_QH}: {len(df_filtered)}')

pareto_mask = paretoset(df_filtered[["BdotN", "max_max_curvature"]], sense=[min, min])
df_pareto = df_filtered[pareto_mask]

print(f"Best Pareto-optimal results (total of {len(df_pareto)}):")
print(
    df_pareto[
        [
            "directory",
            "Jf",
            "max_max_curvature",
            "length",
            "coil_coil_distance",
            "BdotN",
            "order",
        ]
    ]
)
print("Directory names only:")
for dirname, currents in zip(df_pareto["directory"], df_pareto["coil_currents"]):
    print(f'dirname = {dirname}')
    if print_coil_currents: print(f'  currents = {currents}')

#########################################################
# Plotting
#########################################################

os.chdir(os.path.join(results_path,'..'))

plt.figure(figsize=(14.5, 8))
plt.rc("font", size=8)
nrows = 3
ncols = 6
markersize = 5

subplot_index = 1
plt.subplot(nrows, ncols, subplot_index)
subplot_index += 1
plt.scatter(df["Jf"], df["max_max_curvature"], c=df["length"], s=1)
plt.colorbar(label="length")
plt.scatter(
    df_filtered["Jf"],
    df_filtered["max_max_curvature"],
    c=df_filtered["length"],
    s=markersize,
)
plt.scatter(
    df_pareto["Jf"], df_pareto["max_max_curvature"], c=df_pareto["length"], marker="+"
)
plt.xlabel("Bnormal objective")
plt.ylabel("Max curvature")
plt.xscale("log")

plt.subplot(nrows, ncols, subplot_index)
subplot_index += 1
plt.scatter(
    df_filtered["length_target"],
    df_filtered["length"],
    c=df_filtered["Jf"],
    s=markersize,
    norm=colors.LogNorm(),
)
plt.colorbar(label="Bnormal objective")
plt.xlabel("length_target")
plt.ylabel("length")

plt.subplot(nrows, ncols, subplot_index)
subplot_index += 1
plt.scatter(
    df_filtered["max_curvature_threshold"],
    df_filtered["max_max_curvature"],
    c=df_filtered["Jf"],
    s=markersize,
    norm=colors.LogNorm(),
)
plt.colorbar(label="Bnormal objective")
plt.xlabel("max_curvature_threshold")
plt.ylabel("max_max_curvature")

def plot_2d_hist(field, log=False):
    global subplot_index
    plt.subplot(nrows, ncols, subplot_index)
    subplot_index += 1
    nbins = 20
    if log:
        data = df[field]
        bins = np.logspace(np.log10(data.min()), np.log10(data.max()), nbins)
    else:
        bins = nbins
    plt.hist(df[field], bins=bins, label="before filtering")
    plt.hist(df_filtered[field], bins=bins, alpha=1, label="after filtering")
    plt.xlabel(field)
    plt.legend(loc=0, fontsize=6)
    if log:
        plt.xscale("log")

# 2nd entry of each tuple is True if the field should be plotted on a log x-scale.
fields = (
    ("R1", False),
    ("order", False),
    ("length", False),
    ("length_target", False),
    ("length_weight", True),
    ("max_curvature_threshold", False),
    ("max_curvature_weight", True),
    ("max_max_curvature", False),
    ("coil_coil_distance", False),
    ("cc_threshold", False),
    ("cc_weight", True),
)

for field, log in fields:
    plot_2d_hist(field, log)

plt.figtext(0.5, 0.995, os.path.abspath(__file__), ha="center", va="top", fontsize=6)
plt.tight_layout()
# plt.show()
plt.savefig('coil_scan.png', dpi=300)

nfp = 3 
nphi_big = nphi * 2 * nfp + 1
ntheta_big = ntheta + 1
quadpoints_theta = np.linspace(0, 1, ntheta_big)
quadpoints_phi = np.linspace(0, 1, nphi_big)
filename = os.path.join(this_path ,'input.' + QA_or_QH)
surf = SurfaceRZFourier.from_vmec_input(filename, range="half period", nphi=nphi, ntheta=ntheta)
surf_big = SurfaceRZFourier(dofs=surf.dofs,nfp=nfp, mpol=surf.mpol,ntor=surf.ntor,
                            quadpoints_phi=quadpoints_phi,quadpoints_theta=quadpoints_theta)

def process_surface_and_flux(bs, surf, surf_big=None, new_OUT_DIR="", prefix=""):
    bs.set_points(surf.gamma().reshape((-1, 3)))
    Bbs = bs.B().reshape((nphi, ntheta, 3))
    BdotN = (np.sum(Bbs * surf.unitnormal(), axis=2)) / np.linalg.norm(Bbs, axis=2)
    pointData = {"B.n/B": BdotN[:, :, None]}
    surf.to_vtk(os.path.join(new_OUT_DIR, prefix + "halfnfp"), extra_data=pointData)
    if surf_big is not None:
        bs.set_points(surf_big.gamma().reshape((-1, 3)))
        Bbs = bs.B().reshape((nphi_big, ntheta_big, 3))
        BdotN = (np.sum(Bbs * surf_big.unitnormal(), axis=2)) / np.linalg.norm(Bbs, axis=2)
        pointData = {"B.n/B": BdotN[:, :, None]}
        surf_big.to_vtk(os.path.join(new_OUT_DIR, prefix + "big"), extra_data=pointData)

# Copy the best results to a separate directory
optimal_coils_path = os.path.join(results_path, "..", "optimal_coils")
Path(optimal_coils_path).mkdir(parents=True, exist_ok=True)
if os.path.exists(optimal_coils_path):
    shutil.rmtree(optimal_coils_path)
for dirname in df_pareto["directory"]:
    source_dir = os.path.join(results_path, dirname)
    destination_dir = os.path.join(optimal_coils_path, dirname)
    shutil.copytree(source_dir, destination_dir, dirs_exist_ok=True)
    bs=load(os.path.join(source_dir,"biot_savart.json"))
    process_surface_and_flux(bs, surf, surf_big=surf_big, new_OUT_DIR=destination_dir, prefix='surf_opt_')