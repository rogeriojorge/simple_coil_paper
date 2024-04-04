#!/usr/bin/env python3

# This script runs coil optimizations, one after another, choosing the weights
# and target values from a random distribution. This is effectively a crude form
# of global optimization.

import os
import json
import argparse
import numpy as np
from pathlib import Path
from scipy.optimize import minimize
from simsopt.field import BiotSavart, Current, coils_via_symmetries
from simsopt.geo import (curves_to_vtk, create_equally_spaced_curves, SurfaceRZFourier,
                        LinkingNumber, CurveLength, CurveCurveDistance,
                        MeanSquaredCurvature, LpCurveCurvature, create_equally_spaced_planar_curves)
from simsopt.objectives import SquaredFlux, QuadraticPenalty
this_path = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("--type", type=int, default=1)
parser.add_argument("--ncoils", type=int, default=2)
args = parser.parse_args()

if args.type == 1: QA_or_QH = 'simple'
elif args.type == 2: QA_or_QH = 'QA'
elif args.type == 3: QA_or_QH = 'QH'
elif args.type == 4: QA_or_QH = 'QI'
elif args.type == 5: QA_or_QH = 'QI_nfp2'
else: raise ValueError('Invalid type')

ncoils = args.ncoils
R1_mean = 0.75
R1_std = 0.15
extend_distance = 0.01
MAXITER = 800
use_nfp3 = True
opt_method = 'L-BFGS-B'
min_length_per_coil = 2.8
max_length_per_coil = 3.8
min_curvature = 10
max_curvature = 30
CC_min = 0.1
CC_max = 0.2
order_min = 3
order_max = 9
nphi = 26
ntheta = 26
nquadpoints = 80

results_path = os.path.join(os.path.dirname(__file__), 'results_'+QA_or_QH)
Path(results_path).mkdir(parents=True, exist_ok=True)
os.chdir(results_path)
filename = os.path.join(results_path, '..', 'input.' + QA_or_QH)
surf = SurfaceRZFourier.from_vmec_input(filename, range="half period", nphi=nphi, ntheta=ntheta)
nfp = surf.nfp
R0 = surf.get_rc(0, 0)

nphi_big = nphi * 2 * nfp + 1
ntheta_big = ntheta + 1
quadpoints_theta = np.linspace(0, 1, ntheta_big)
quadpoints_phi = np.linspace(0, 1, nphi_big)

out_dir = os.path.join(results_path,"scan")
os.makedirs(out_dir, exist_ok=True)
os.chdir(out_dir)

surf_big = SurfaceRZFourier(dofs=surf.dofs,nfp=nfp, mpol=surf.mpol,ntor=surf.ntor,
                            quadpoints_phi=quadpoints_phi,quadpoints_theta=quadpoints_theta)

def run_optimization(
    R1,
    order,
    length_target,
    length_weight,
    max_curvature_threshold,
    max_curvature_weight,
    msc_threshold,
    msc_weight,
    cc_threshold,
    cc_weight,
    index,
):
    directory = (
        f"ncoils_{ncoils}_order_{order}_R1_{R1:.2}_length_target_{length_target:.2}_weight_{length_weight:.2}"
        + f"_max_curvature_{max_curvature_threshold:.2}_weight_{max_curvature_weight:.2}"
        + f"_msc_{msc_threshold:.2}_weight_{msc_weight:.2}"
        + f"_cc_{cc_threshold:.2}_weight_{cc_weight:.2}"
    )

    print()
    print("***********************************************")
    print(f"Job {index+1}")
    print("Parameters:", directory)
    print("***********************************************")
    print()

    # Directory for output
    new_OUT_DIR = directory + "/"
    os.mkdir(directory)

    # Create the initial coils:
    # base_curves = create_equally_spaced_curves(ncoils, nfp, stellsym=True, R0=R0, R1=R1, order=order, numquadpoints=nquadpoints)
    base_curves = create_equally_spaced_planar_curves(ncoils, surf.nfp, stellsym=True, R0=R0, R1=R1, order=order, numquadpoints=nquadpoints)
    
    def process_surface_and_flux(bs, surf, surf_big=None, new_OUT_DIR="", prefix=""):
        bs.set_points(surf.gamma().reshape((-1, 3)))
        Bbs = bs.B().reshape((nphi, ntheta, 3))
        BdotN = (np.sum(Bbs * surf.unitnormal(), axis=2)) / np.linalg.norm(Bbs, axis=2)
        maxBdotN = np.max(np.abs(BdotN))
        pointData = {"B.n/B": BdotN[:, :, None]}
        surf.to_vtk(new_OUT_DIR + prefix + "halfnfp", extra_data=pointData)
        if surf_big is not None:
            bs.set_points(surf_big.gamma().reshape((-1, 3)))
            Bbs = bs.B().reshape((nphi_big, ntheta_big, 3))
            BdotN = (np.sum(Bbs * surf_big.unitnormal(), axis=2)) / np.linalg.norm(Bbs, axis=2)
            pointData = {"B.n/B": BdotN[:, :, None]}
            surf_big.to_vtk(new_OUT_DIR + prefix + "big", extra_data=pointData)
        bs.set_points(surf.gamma().reshape((-1, 3)))
        Jf = SquaredFlux(surf, bs, definition="local")
        return Jf, maxBdotN
    
    base_currents = [Current(1.0) * (1e5) for i in range(ncoils)]
    # base_currents[0].fix_all()
    coils = coils_via_symmetries(base_curves, base_currents, nfp, True)
    curves = [c.curve for c in coils]
    curves_to_vtk(curves, new_OUT_DIR + "curves_init", close=True)
    bs = BiotSavart(coils)
    Jf_total, _ = process_surface_and_flux(bs, surf, surf_big=surf_big, new_OUT_DIR=new_OUT_DIR, prefix='surf_init_')

    Jls = [CurveLength(c) for c in base_curves]
    Jccdist = CurveCurveDistance(curves, cc_threshold, num_basecurves=ncoils)
    Jcs = [LpCurveCurvature(c, 2, max_curvature_threshold) for c in base_curves]
    Jmscs = [MeanSquaredCurvature(c) for c in base_curves]
    JF = (
        Jf_total
        + length_weight * QuadraticPenalty(sum(Jls), length_target * ncoils)
        + cc_weight * Jccdist
        + max_curvature_weight * sum(Jcs)
        + msc_weight * sum(QuadraticPenalty(J, msc_threshold, "max") for J in Jmscs)
        + LinkingNumber(curves, 2)
    )
    
    iteration = 0

    def fun(dofs):
        nonlocal iteration
        JF.x = dofs
        J = JF.J()
        grad = JF.dJ()
        jf = Jf_total.J()
        Bbs = bs.B().reshape((nphi, ntheta, 3))
        BdotN = np.max(np.abs((np.sum(Bbs * surf.unitnormal(), axis=2)) / np.linalg.norm(Bbs, axis=2)))
        outstr = f"{iteration:4} J={J:.1e}, Jf={jf:.1e}"
        outstr += f", max⟨B·n⟩/B={BdotN:.1e}"
        cl_string = ", ".join([f"{J.J():.1f}" for J in Jls])
        kap_string = ", ".join(f"{np.max(c.kappa()):.1f}" for c in base_curves)
        msc_string = ", ".join(f"{J.J():.1f}" for J in Jmscs)
        outstr += f", L=[{cl_string}], "#={sum(J.J() for J in Jls):.1f}, "
        outstr += f"ϰ=[{kap_string}], msc=[{msc_string}]"
        outstr += f", CC={Jccdist.shortest_distance():.2f}"
        # outstr += f", cs1={Jcsdist1.shortest_distance():.2f}"
        # outstr += f", cs2={Jcsdist2.shortest_distance():.2f}"
        # outstr += f", ║∇J║={np.linalg.norm(grad):.1e}"
        print(outstr)
        iteration += 1
        return J, grad

    res = minimize( fun, JF.x, jac=True, method="L-BFGS-B", options={"maxiter": MAXITER, "maxcor": 300}, tol=1e-8)
    JF.x = res.x
    print(res.message)
    curves_to_vtk(curves, new_OUT_DIR + "curves_opt", close=True)
    curves_to_vtk(base_curves, new_OUT_DIR + "curves_opt_halfnfp", close=True)

    Jf_total, BdotN = process_surface_and_flux(bs, surf, surf_big=surf_big, new_OUT_DIR=new_OUT_DIR, prefix='surf_opt_')
    bs.save(new_OUT_DIR + "biot_savart.json")

    # Save the optimized coil shapes and currents so they can be loaded into other scripts for analysis:

    results = {
        "nfp": nfp,
        "R0": R0,
        "R1": R1,
        "ncoils": ncoils,
        "order": order,
        "nphi": nphi,
        "ntheta": ntheta,
        "length_target": length_target,
        "length_weight": length_weight,
        "max_curvature_threshold": max_curvature_threshold,
        "max_curvature_weight": max_curvature_weight,
        "msc_threshold": msc_threshold,
        "msc_weight": msc_weight,
        "JF": float(JF.J()),
        "Jf": float(Jf_total.J()),
        "BdotN": BdotN,
        "lengths": [float(J.J()) for J in Jls],
        "length": float(sum(J.J() for J in Jls)),
        "max_curvatures": [np.max(c.kappa()) for c in base_curves],
        "max_max_curvature": max(np.max(c.kappa()) for c in base_curves),
        "coil_coil_distance": Jccdist.shortest_distance(),
        "cc_threshold": cc_threshold,
        "cc_weight": cc_weight,
        "gradient_norm": np.linalg.norm(JF.dJ()),
        "linking_number": LinkingNumber(curves).J(),
        "directory": directory,
        "mean_squared_curvatures": [float(J.J()) for J in Jmscs],
        "max_mean_squared_curvature": float(max(J.J() for J in Jmscs)),
        "message": res.message,
        "success": res.success,
        "iterations": res.nit,
        "function_evaluations": res.nfev,
        "coil_currents": [c.get_value() for c in base_currents],
    }

    with open(new_OUT_DIR + "results.json", "w") as outfile:
        json.dump(results, outfile, indent=2)


#########################################################################
# Carry out the scan. Below you can adjust the ranges for the random weights and
# thresholds.
#########################################################################


def rand(min, max):
    """Generate a random float between min and max."""
    return np.random.rand() * (max - min) + min


for index in range(10000):
    # Initial radius of the coils:
    R1 = np.random.rand() * R1_std + R1_mean

    # Number of Fourier modes describing each Cartesian component of each coil:
    order = int(np.round(rand(order_min, order_max)))

    # Target length (per coil!) and weight for the length term in the objective function:
    length_target = rand(min_length_per_coil, max_length_per_coil)
    length_weight = 10.0 ** rand(-4, 0)

    # Threshold and weight for the curvature penalty in the objective function:
    max_curvature_threshold = rand(min_curvature, max_curvature)
    max_curvature_weight = 10.0 ** rand(-6, -1)

    # Threshold and weight for the mean squared curvature penalty in the objective function:
    msc_threshold = rand(min_curvature, max_curvature)
    msc_weight = 10.0 ** rand(-6, -2)

    # Threshold and weight for the coil-to-coil distance penalty in the objective function:
    cc_threshold = rand(CC_min, CC_max)
    cc_weight = 10.0 ** rand(-0, 4)

    run_optimization(
        R1,
        order,
        length_target,
        length_weight,
        max_curvature_threshold,
        max_curvature_weight,
        msc_threshold,
        msc_weight,
        cc_threshold,
        cc_weight,
        index,
    )
