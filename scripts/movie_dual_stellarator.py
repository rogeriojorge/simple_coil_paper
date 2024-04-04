#!/usr/bin/env python3
import os
import re
import imageio
import numpy as np
import pyvista as pv
from simsopt import load
from pathlib import Path
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from simsopt.geo import SurfaceRZFourier
from pyvista.utilities import lines_from_points
this_path = '/Users/rogeriojorge/local/dual_stellarator'

# Constants
nphi = 60
ntheta = 60
N_interpolation = 30
N_extra = 2
x_centered = 5
def interp_function(x): return 1/(1+np.exp(-x+x_centered))
interp_factors = sorted(np.concatenate([np.linspace(0,0.03,N_extra,endpoint=True),interp_function(np.linspace(0, x_centered*2, N_interpolation, endpoint=True)),np.linspace(0.97,1.0,N_extra,endpoint=True)]))

## Input parameters
output_path = 'surfs_between_nfp3'
# optimization_folder = 'optimization_good_ncoils_7_order_6_R1_0.45_length_target_2.4_weight_10000.0_max_curvature_19.0_weight_1.6e-06_msc_15.0_weight_1.7e-06_cc_0.0841_weight_3.0_QH_0.03_weight_20000.0'
# input1 = 'input.nfp3_QA_final'
input1 = 'input.QA_final'
optimization_folder = 'optimization_good_ncoils_7_order_5_R1_0.47_length_target_2.2_weight_10000.0_max_curvature_6.5_weight_1.4e-05_msc_14.0_weight_3.1e-05_cc_0.0758_weight_1100.0_QH_0.03_weight_20000.0'

## Load surfaces and coils
common_part = input1.split('.')[1]
input2 = f'input.{common_part.replace("QA", "QH")}'
biot_savart1 = f'biot_savart_{common_part}.json'
biot_savart2 = f'biot_savart_{common_part.replace("QA", "QH")}.json'
curves_file = f'base_curves_{common_part}.vtu'

results_path = os.path.join(this_path, optimization_folder)
os.chdir(results_path)
coils_path = os.path.join(results_path, 'coils')
filename1 = os.path.join(results_path, input1)
filename2 = os.path.join(results_path, input2)
ncoils = int(re.search(r'ncoils_(\d+)', optimization_folder).group(1))

# Load surfaces
surf1 = SurfaceRZFourier.from_vmec_input(filename1, range="full torus", nphi=nphi, ntheta=ntheta)
surf2 = SurfaceRZFourier.from_vmec_input(filename2, range="full torus", nphi=nphi, ntheta=ntheta)
surf_between = SurfaceRZFourier.from_vmec_input(filename1, range="full torus", nphi=nphi, ntheta=ntheta)
surf1_dofs = surf1.x
surf2_dofs = surf2.x

# Load coils
bs1 = load(os.path.join(coils_path, biot_savart1))
bs2 = load(os.path.join(coils_path, biot_savart2))
bs1_dofs = bs1.x
bs2_dofs = bs2.x
currents_1 = np.array([bs1.coils[i].current.get_value() for i in range(ncoils)])
currents_2 = np.array([bs2.coils[i].current.get_value() for i in range(ncoils)])
max_current = max(max(currents_1), max(currents_2))
min_current = min(min(currents_1), min(currents_2))/max_current
print(f"Coil currents 1: {currents_1}")
print(f"Coil currents 2: {currents_2}")

# Bar plot
plt.figure(figsize=(6, 4))  # Adjust figure size
bar_width = 0.35
index = np.arange(len(currents_1))

plt.bar(index, currents_1/np.max(currents_1), bar_width, color='b', label=r'Coil currents 1')
plt.bar(index + bar_width, currents_2/np.max(currents_1), bar_width, color='r', label=r'Coil currents 2')

plt.xlabel('Coil', fontsize=14)  # Increase font size
plt.ylabel('Current Values', fontsize=14)  # Increase font size
plt.title('Coil Currents', fontsize=16)  # Increase font size
plt.xticks(index + bar_width / 2, range(1, len(currents_1) + 1), fontsize=14)  # Increase font size and adjust ticks
plt.yticks(fontsize=14)  # Increase font size for y ticks
plt.legend(fontsize=11)  # Increase font size for legend
plt.tight_layout()
plt.savefig(os.path.join(output_path, '..', 'coil_currents_histogram.png'), dpi=300)  # Adjust DPI for higher resolution

# Read coils mesh
coils_vtu = pv.read(os.path.join(coils_path, curves_file))

# Create output directory if not exists
Path(output_path).mkdir(parents=True, exist_ok=True)

# Create a list to store the file names of the PNG images
image_files = []

if len(bs1_dofs)<len(bs2_dofs): bs1_dofs = np.concatenate([[1.0],bs1_dofs])
else:                           bs2_dofs = np.insert(bs2_dofs,5,1.0)
# Convert VTK files to PNG images and store file names
clim=0
num_frames = 2 * len(interp_factors) - 1
save_indices = [0, num_frames // 2, num_frames - 1]  # Indices to save the PNGs
for i in range(2 * len(interp_factors) - 1):
    j = 2 * len(interp_factors) - i - 2 if i > len(interp_factors) - 1 else i
    factor = interp_factors[j]
    # print(factor)

    # Interpolate surfaces and currents
    surf_between.x = (1 - factor) * surf1_dofs + factor * surf2_dofs
    bs1.x = (1 - factor) * bs1_dofs + factor * bs2_dofs
    bs1.set_points(surf_between.gamma().reshape((-1, 3)))

    # Calculate surface normals
    BdotN1 = (np.sum(bs1.B().reshape((nphi, ntheta, 3)) * surf_between.unitnormal(), axis=2)) / np.linalg.norm(bs1.B().reshape((nphi, ntheta, 3)), axis=2)
    print(f'max BdotN: {np.max(np.abs(BdotN1))}')
    if clim==0: clim=np.max(np.abs(BdotN1))
    surf_between.to_vtk(os.path.join(output_path, f"surf_between_halfnfp_{i}"), extra_data={"B.n/B": BdotN1[:, :, None]})

    # Plot surfaces and coils
    vtk_file = os.path.join(output_path, f"surf_between_halfnfp_{j}.vts")
    png_file = os.path.join(output_path, f"surf_between_halfnfp_{j}.png")
    surf_between_vtk = pv.read(vtk_file)

    plotter = pv.Plotter(off_screen=True)

    args_cbar = dict(height=0.1, vertical=False, position_x=0.22, position_y=0.03, color="k", title_font_size=24, label_font_size=16)

    surf_mesh = plotter.add_mesh(surf_between_vtk, scalars="B.n/B", cmap="coolwarm", clim=[-clim, clim], scalar_bar_args=args_cbar)
    # Normalize current values
    current_values = np.array([coil.current.get_value() for coil in bs1.coils])
    these_currents = []
    for coil_index, coil in enumerate(bs1.coils):
        # Extract the points for the current coil
        coil_points = coils_vtu.extract_cells(coil_index)

        # Check if the coil has points before plotting
        if coil_points.n_points > 0:
            args_cbar = dict(width=0.05, vertical=True, position_x=0.03, position_y=0.03, color="k", title_font_size=24, label_font_size=16, title='Current')
            plotter.add_mesh(coil_points, line_width=6, show_edges=True, label=f"Coil {coil_index}", scalars=[current_values[coil_index]/max_current]*coil_points.n_points, cmap="coolwarm", scalar_bar_args=args_cbar, clim=[-1,1])#, clim=[min_current, 1])
            
            # Add text annotations for coil indices
            coil_center = coil_points.center
            this_current = current_values[coil_index]/max_current
            coil_numbers = ["6", "7", "5", "4", "3", "2", "1"]
            if i in [0,len(interp_factors)]: print(f'coil {coil_numbers[coil_index]} current: {this_current}')# at {coil_center}')
            plotter.add_text(f"{coil_numbers[coil_index]}", position=[510-coil_index*50,670-coil_index*16], font_size=25, color="black")
            these_currents.append(this_current)

    # plotter.add_axes()
    # # Add scalar bar for the current values legend
    # plotter.add_scalar_bar(title="Currents", vertical=True, width=0.05, position_x=0.05, position_y=0.1, title_font_size=24, label_font_size=16)

    # Set background to white
    plotter.set_background("white")
    
    # Adjust camera position (rotate and zoom)
    plotter.camera_position = (-5.1,-0.9,3)#[(7, -2, 0), (0, 0, 0), (0, 0, 1)]  # Example camera position (adjust as needed)
    # plotter.camera_clipping_range = [5, 20]  # Adjust the clipping range to zoom in
    plotter.camera.SetFocalPoint([0, 0, 0])  # Set focal point at the center of the scene
    plotter.camera.zoom((1 - factor) * 1.6 + factor * 1.48)

    f, ax = plt.subplots(tight_layout=True)  # Adjust figure size
    bar_width = 0.35
    index_1 = [7,6,5,4,3,2,1]#np.array([float(number) for number in coil_numbers])
    index_2 = np.arange(0.8,len(currents_1)+0.8)
    cmap = cm.coolwarm
    # colors = plt.cm.coolwarm(these_currents)
    vmin, vmax = -1, 1
    colors = [cmap((current - vmin) / (vmax - vmin)) for current in these_currents]
    plt.bar(index_1, these_currents, bar_width, color=colors, label=r'Coil Currents')
    plt.xlabel('Coil', fontsize=14)  # Increase font size
    plt.ylabel('Currents', fontsize=14)  # Increase font size
    # plt.title('Coil Currents', fontsize=16)  # Increase font size
    plt.xticks(index_2 + bar_width / 2, range(1, len(currents_1) + 1), fontsize=14)  # Increase font size and adjust ticks
    plt.yticks(fontsize=14)  # Increase font size for y ticks
    plt.ylim([-1,1])
    # plt.legend(fontsize=11)  # Increase font size for legend

    h_chart = pv.ChartMPL(f, size=(0.39, 0.23), loc=(0.58, 0.7))
    h_chart.background_color = (1.0, 1.0, 1.0, 0.4)
    plotter.add_chart(h_chart)
    plt.close(f)

    plotter.show(screenshot=png_file)
    image_files.append(png_file)
    # exit()
    
# Create a gif from the PNG images
gif_file = os.path.join(output_path, "surf_between_animation.gif")
with imageio.get_writer(gif_file, mode='I') as writer:
    for image_file in image_files:
        image = imageio.v2.imread(image_file)
        writer.append_data(image)

# Print the path to the generated gif
print(f"GIF created: {gif_file}")

# Remove the VTS and PNG files except for the saved ones
for vtk_file in os.listdir(output_path):
    if vtk_file.endswith(".vts"):
        os.remove(os.path.join(output_path, vtk_file))
    if vtk_file.endswith(".png"):
        if vtk_file not in [f"surf_between_halfnfp_{index}.png" for index in save_indices]:
            os.remove(os.path.join(output_path, vtk_file))