#!/usr/bin/env python3
import os
import shutil
import argparse
import numpy as np
from pathlib import Path
from functools import partial
from scipy.optimize import minimize
from simsopt import make_optimizable
from simsopt._core.derivative import Derivative
from simsopt.solve import least_squares_mpi_solve
from simsopt.mhd import Vmec, QuasisymmetryRatioResidual
from simsopt.field.magneticfieldclasses import ToroidalField
from simsopt.util import MpiPartition, proc0_print, comm_world
from simsopt._core.finite_difference import MPIFiniteDifference
from simsopt.field import BiotSavart, Current, coils_via_symmetries, Coil
from simsopt.objectives import SquaredFlux, QuadraticPenalty, LeastSquaresProblem
from simsopt.geo import (CurveLength, CurveCurveDistance, MeanSquaredCurvature, SurfaceRZFourier, create_equally_spaced_planar_curves, CurveSurfaceDistance,
                         LpCurveCurvature, ArclengthVariation, curves_to_vtk, create_equally_spaced_curves, LinkingNumber, CurveHelical)
from simsopt.geo.curvexyzfouriersymmetries import CurveXYZFourierSymmetries
from simsopt.geo.curvexyzfourier import CurveXYZFourier
from simsopt.geo.curve import RotatedCurve
from simsopt.field.coil import ScaledCurrent

mpi = MpiPartition()
parent_path = str(Path(__file__).parent.resolve())
os.chdir(parent_path)
parser = argparse.ArgumentParser()
parser.add_argument("--type", type=int, default=2)
parser.add_argument("--ncoils", type=int, default=1)
parser.add_argument("--extra", type=int, default=2)
parser.add_argument("--symcoils", type=int, default=2)
parser.add_argument("--l0", type=int, default=6)
args = parser.parse_args()
if args.type == 1: QA_or_QH = 'simple_nfp1'
elif args.type == 2: QA_or_QH = 'QA'
elif args.type == 3: QA_or_QH = 'QH'
elif args.type == 4: QA_or_QH = 'QI'
elif args.type == 5: QA_or_QH = 'simple_nfp2'
elif args.type == 6: QA_or_QH = 'simple_nfp3'
elif args.type == 7: QA_or_QH = 'simple_nfp4'
elif args.type == 8: QA_or_QH = 'simple_nfp5'
elif args.type == 9: QA_or_QH = 'simple_nfp6'
elif args.type == 10: QA_or_QH = 'simple_nfp4_planar'
elif args.type == 11: QA_or_QH = 'simple_nfp3_planar'
elif args.type == 12: QA_or_QH = 'QI_nfp2'
else: raise ValueError('Invalid type')
ncoils = args.ncoils
##########################################################################################
############## Input parameters
##########################################################################################
optimize_stage_1_with_coils = False
if args.symcoils==1: stellsym_coils = True
else: stellsym_coils = False
if args.extra==1: use_extra_coils = True
else:             use_extra_coils = False
MAXITER_stage_1 = 20
MAXITER_stage_2 = 100 # 300
# MAXITER_single_stage = 25
# MAXFEV_single_stage  = 42
max_mode_array = [1]*1 + [2]*1 + [3]*1 + [4]*2 + [5]*0 + [6]*0
# ncoils = 1
l0_coil = args.l0
order_coils = l0_coil*3+2
if ncoils==0:
    use_two_coils = True
    LENGTH_THRESHOLD = 12
    nquadpoints = 350
else:
    LENGTH_THRESHOLD = 10.0*l0_coil if QA_or_QH=='QA' else (12.0*l0_coil if l0_coil>4 else 13.5*l0_coil)
    nquadpoints = int(LENGTH_THRESHOLD*26)
ro_coil = 0.6
aspect_ratio_target = 8.5
JACOBIAN_THRESHOLD = 300 if ncoils >0 else 2e2
aspect_ratio_weight = 1e+1 # 3e-2 if 'QA' in QA_or_QH else (8e-3 if 'QI' in QA_or_QH else (4e-2 if QA_or_QH=='simple_nfp4' else (3e-2 if QA_or_QH=='simple_nfp3' else 2e-2)))
nfp_min_iota_nfp4 = 0.252; nfp_min_iota_nfp3 = 0.175; nfp_min_iota = 0.11; nfp_min_iota_QH = 0.65; nfp_min_iota_QA = 0.41
iota_min_QA = nfp_min_iota_QA if QA_or_QH=='QA' else (nfp_min_iota_nfp4 if QA_or_QH=='simple_nfp4' else (nfp_min_iota_nfp3 if QA_or_QH=='simple_nfp3' else nfp_min_iota))
iota_min_QH = nfp_min_iota_QH if QA_or_QH=='QH' else (nfp_min_iota_nfp4 if QA_or_QH=='simple_nfp4' else (nfp_min_iota_nfp3 if QA_or_QH=='simple_nfp3' else nfp_min_iota))
maxmodes_mpol_mapping = {1: 5, 2: 5, 3: 5, 4: 5, 5: 6, 6: 7}
MAXITER_single_stage_mpol_mapping = {1: 22, 2: 25, 3: 30, 4: 40}
MAXFEV_single_stage_mpol_mapping  = {1: 30, 2: 35, 3: 40, 4: 60} if ncoils>0 else {1: 15, 2: 15, 3: 20, 4: 25}
quasisymmetry_weight_mpol_mapping = {1: 1e-2, 2: 1e-2, 3: 1e-2, 4: 1e-2} if ncoils==0 else ({1: 2e+1, 2: 7e+1, 3: 1e+2, 4: 2e+2} if QA_or_QH=='QA' else {1: 3e+2, 2: 8e+2, 3: 3e+3, 4: 1e+4})
coils_objective_weight = 1e+3 if 'QI' in QA_or_QH else 5e+4
CC_THRESHOLD = 0.12
CS_THRESHOLD = 0.02
CS_WEIGHT = 1e5
# quasisymmetry_weight = 5e+4 if QA_or_QH=='QA' else 2e+2 # 1e-0 if 'QI' in QA_or_QH else 1e+2
# QA_or_QH = 'simple' # QA, QH, QI or simple
vmec_input_filename = os.path.join(parent_path, 'input.'+ QA_or_QH)
CURVATURE_THRESHOLD = 10 if ncoils>1 else 4
MSC_THRESHOLD = 10 if ncoils>1 else 4
nphi_VMEC = 128 if use_extra_coils else (32 if stellsym_coils else 64)
ntheta_VMEC = 32
ftol = 1e-3
diff_method = "forward"
R0 = 1.0
R1 = 0.58 if 'QA' in QA_or_QH else 0.7
mirror_weight = 1e+4
maximum_mirror = 0.21 if 'QI' in QA_or_QH else 0.23
weight_iota = 1e3
elongation_weight = 1
directory = f'optimization_{QA_or_QH}'
if stellsym_coils: directory += '_symcoils'
else: directory += '_asymcoils'
if use_extra_coils: directory += '_extracoils'
directory += f'_l0{l0_coil}' if ncoils>0 else '_QAlinked'
quasisymmetry_target_surfaces = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
finite_difference_abs_step = 1e-7
finite_difference_rel_step = 1e-4
LENGTH_CON_WEIGHT = 5.0e-2  # Weight on the quadratic penalty for the curve length
CC_WEIGHT = 3.6e+2  # Weight for the coil-to-coil distance penalty in the objective function
CURVATURE_WEIGHT = 1e-2  # Weight for the curvature penalty in the objective function
MSC_WEIGHT = 2.8e-5 # Weight for the mean squared curvature penalty in the objective function
ARCLENGTH_WEIGHT = 1e-9  # Weight for the arclength variation penalty in the objective function
##########################################################################################
##########################################################################################
vmec_verbose = False
# Create output directories
this_path = os.path.join(parent_path, directory)
os.makedirs(this_path, exist_ok=True)
shutil.copyfile(os.path.join(parent_path, 'main.py'), os.path.join(this_path, 'main.py'))
os.chdir(this_path)
vmec_results_path = os.path.join(this_path, "vmec")
coils_results_path = os.path.join(this_path, "coils")
if comm_world.rank == 0:
    os.makedirs(vmec_results_path, exist_ok=True)
    os.makedirs(coils_results_path, exist_ok=True)
##########################################################################################
##########################################################################################
# Stage 1
proc0_print(f' Using vmec input file {vmec_input_filename}')
vmec = Vmec(vmec_input_filename, mpi=mpi, verbose=vmec_verbose, nphi=nphi_VMEC, ntheta=ntheta_VMEC, range_surface='full torus' if use_extra_coils else ('half period' if stellsym_coils else 'field period'))
surf = vmec.boundary
nphi_big   = nphi_VMEC if use_extra_coils else nphi_VMEC * 2 * surf.nfp + 1
ntheta_big = ntheta_VMEC + 1
quadpoints_theta = np.linspace(0, 1, ntheta_big)
quadpoints_phi   = np.linspace(0, 1, nphi_big)
surf_big = SurfaceRZFourier(dofs=surf.dofs, nfp=surf.nfp, mpol=surf.mpol, ntor=surf.ntor, quadpoints_phi=quadpoints_phi, quadpoints_theta=quadpoints_theta, stellsym=surf.stellsym)
##########################################################################################
##########################################################################################
#Stage 2
if ncoils == 0:
    order_coils = 6
    extra_coil_R1 = 1.5
    base_currents = [1.00e5, -1.00e5]
    if surf.nfp != 2:
        raise ValueError('Invalid surface configuration, only for nfp=2 QA')
    base_curves = [CurveXYZFourierSymmetries(np.linspace(0, 1, nquadpoints, endpoint=False), order_coils, surf.nfp, surf.stellsym)]
    base_curves[0].x = [1.096374667510205,0.4381494495775816,0.022334134293422507,
                        -1.3497221392404835e-17,1.0502404453327285e-17,-3.029486747096698e-18,
                        -1.5890569401485296e-17,-0.2148081066433566,-0.02233413429342252,-3.583155306349866e-17,
                        -3.5434233836070224e-18,-2.6416320204511416e-17,4.309936587795308e-18,
                        0.40734354634861325,-3.1146068220063536e-17,9.386440688229218e-18,
                        -3.6326617164643065e-17,-3.214035238347376e-17,3.2453413828866613e-17
                    ]
    coils = [Coil(base_curves[0], Current(1)*base_currents[0])]
    if use_two_coils:
        base_curves += [CurveXYZFourierSymmetries(np.linspace(0, 1, nquadpoints, endpoint=False), order_coils, surf.nfp, surf.stellsym)]
        base_curves[1].x = [1.096374667510205,-0.5181494495775816,0.022334134293422507,
                        -1.3497221392404835e-17,1.0502404453327285e-17,-3.029486747096698e-18,
                        -1.5890569401485296e-17,0.4348081066433566,-0.02233413429342252,-3.583155306349866e-17,
                        -3.5434233836070224e-18,-2.6416320204511416e-17,4.309936587795308e-18,
                        0.50734354634861325,-3.1146068220063536e-17,9.386440688229218e-18,
                        -3.6326617164643065e-17,-3.214035238347376e-17,3.2453413828866613e-17
                    ]
        coils += [Coil(base_curves[1], Current(1)*base_currents[1])]
    # coils = [Coil(base_curves[0], Current(1)*1e5)]
elif ncoils == 1:
    base_curves = [CurveXYZFourierSymmetries(np.linspace(0, 1, nquadpoints, endpoint=False), order_coils, surf.nfp, surf.stellsym)]
    base_curves[0].set('xc(0)', 1)
    base_curves[0].set(f'xc({int(l0_coil)})', ro_coil)
    base_curves[0].set(f'zs({int(l0_coil)})', ro_coil)
    coils = [Coil(base_curves[0], Current(1)*1e5)]
elif ncoils==2:
    base_curves = [CurveXYZFourierSymmetries(np.linspace(0, 1, nquadpoints, endpoint=False), order_coils, surf.nfp, surf.stellsym),
                   CurveXYZFourierSymmetries(np.linspace(0, 1, nquadpoints, endpoint=False), order_coils, surf.nfp, surf.stellsym)]
    base_curves[1].set('xc(0)', 1)
    base_curves[1].set(f'xc({int(l0_coil)})', 1.4*ro_coil)
    base_curves[1].set(f'zs({int(l0_coil)})', 1.4*ro_coil)
    base_curves[0].set('xc(0)', 1)
    base_curves[0].set(f'xc({int(l0_coil)})', ro_coil)
    base_curves[0].set(f'zs({int(l0_coil)})', ro_coil)
    coils = [Coil(base_curves[0], Current(1)*1e5),
            Coil(base_curves[1], Current(-1)*1e5)]
else:
    raise ValueError('Invalid number of coils')
# base_curves = [CurveHelical(nquadpoints, order_coils, surf.nfp, l0_coil, 1., ro_coil) for i in range(2)]
# base_curves[0].set_dofs(np.concatenate((coils_dofs_1 + [0]*(order_coils-2), [0]*(order_coils))))
# base_curves[1].set_dofs(np.concatenate(([0]*(order_coils), coil_dofs_2 + [0]*(order_coils-2))))
# base_currents = [3.07e5, -3.07e5]
# coils = [Coil(base_curves[0], Current(base_currents[0])),
#          Coil(base_curves[1], Current(base_currents[1]))]
Btoroidal = ToroidalField(1.0, 1.0)

if use_extra_coils:
    order = 5
    radius1_array = [0.6]#, 0.6]#, 0.6]#, 0.6]
    center1_array = [0.85]#, 0.95]#, 0.8]#, 0.8]
    y1_array = [-1.2]#, 1.3]#, 1.6]#, -1.6]
    for count, (radius1, center1, y1) in enumerate(zip(radius1_array, center1_array, y1_array)):
        new_base_curve = CurveXYZFourier(128, order)
        new_base_current = Current(-1) * 1e5
        new_base_curve.set_dofs(np.concatenate(([       0, 0, radius1],np.zeros(2*(order-1)),[y1,                radius1, 0],np.zeros(2*(order-1)),[-center1,                     0, 0],np.zeros(2*(order-1)))))
        base_curves += [new_base_curve]
        coils += coils_via_symmetries([new_base_curve], [new_base_current], surf.nfp, stellsym=stellsym_coils)

curves = [c.curve for c in coils]
bs = BiotSavart(coils)#+Btoroidal
##########################################################################################
##########################################################################################
# Save initial surface and coil data
bs.set_points(surf.gamma().reshape((-1, 3)))
Bbs = bs.B().reshape((nphi_VMEC, ntheta_VMEC, 3))
BdotN_surf = np.sum(Bbs * surf.unitnormal(), axis=2) / np.linalg.norm(Bbs, axis=2)
if comm_world.rank == 0:
    curves_to_vtk(curves, os.path.join(coils_results_path, "curves_init"))
    pointData = {"B.n/B": BdotN_surf[:, :, None]}
    surf.to_vtk(os.path.join(coils_results_path, "surf_init"), extra_data=pointData)
bs.set_points(surf_big.gamma().reshape((-1, 3)))
Bbs = bs.B().reshape((nphi_big, ntheta_big, 3))
BdotN_surf = np.sum(Bbs * surf_big.unitnormal(), axis=2) / np.linalg.norm(Bbs, axis=2)
if comm_world.rank == 0:
    pointData = {"B.n/B": BdotN_surf[:, :, None]}
    surf_big.to_vtk(os.path.join(coils_results_path, "surf_init_big"), extra_data=pointData)
bs.set_points(surf.gamma().reshape((-1, 3)))
##########################################################################################
##########################################################################################
Jf = SquaredFlux(surf, bs, definition="local")
Jls = [CurveLength(c) for c in base_curves]
Jccdist = CurveCurveDistance(curves, CC_THRESHOLD, num_basecurves=len(curves))
Jcs = [LpCurveCurvature(c, 2, CURVATURE_THRESHOLD) for i, c in enumerate(base_curves)]
Jmscs = [MeanSquaredCurvature(c) for c in base_curves]
Jals = [ArclengthVariation(c) for c in base_curves]
Jcsdist = CurveSurfaceDistance(curves, surf, CS_THRESHOLD)
# J_LENGTH = LENGTH_WEIGHT * sum(Jls)
J_CC = CC_WEIGHT * Jccdist
J_CURVATURE = CURVATURE_WEIGHT * sum(Jcs)
J_MSC = MSC_WEIGHT * sum(QuadraticPenalty(J, MSC_THRESHOLD, "max") for J in Jmscs)
J_ALS = ARCLENGTH_WEIGHT * sum(Jals)
J_LENGTH_PENALTY = LENGTH_CON_WEIGHT * sum(QuadraticPenalty(J, LENGTH_THRESHOLD, "max") for J in Jls)
linkNum = LinkingNumber(curves)
J_CS = CS_WEIGHT * Jcsdist
JF = Jf + J_LENGTH_PENALTY + J_CURVATURE + J_MSC + J_ALS + J_CC + J_CS# + linkNum
##########################################################################################
proc0_print('  Starting optimization')
##########################################################################################
# Initial stage 2 optimization
def MirrorRatioPen(vmec,t=0.21):
    vmec.run()
    xm_nyq = vmec.wout.xm_nyq
    xn_nyq = vmec.wout.xn_nyq
    bmnc = vmec.wout.bmnc.T
    bmns = 0*bmnc
    nfp = vmec.wout.nfp
    Ntheta = 300
    Nphi = 300
    thetas = np.linspace(0,2*np.pi,Ntheta)
    phis = np.linspace(0,2*np.pi/nfp,Nphi)
    phis2D,thetas2D=np.meshgrid(phis,thetas)
    b = np.zeros([Ntheta,Nphi])
    for imode in range(len(xn_nyq)):
        angles = xm_nyq[imode]*thetas2D - xn_nyq[imode]*phis2D
        b += bmnc[1,imode]*np.cos(angles) + bmns[1,imode]*np.sin(angles)
    Bmax = np.max(b)
    Bmin = np.min(b)
    m = (Bmax-Bmin)/(Bmax+Bmin)
    # print("Mirror =",m)
    pen = np.max([0,m-t])
    return pen
partial_MirrorRatioPen = partial(MirrorRatioPen,t=maximum_mirror)
##########################################################################################
## The function fun_coils defined below is used to only optimize the coils at the beginning
## and then optimize the coils and the surface together. This makes the overall optimization
## more efficient as the number of iterations needed to achieve a good solution is reduced.
def fun_coils(dofss, info):
    info['Nfeval'] += 1
    JF.x = dofss
    J = JF.J()
    grad = JF.dJ()
    if mpi.proc0_world:
        jf = Jf.J()
        Bbs = bs.B().reshape((nphi_VMEC, ntheta_VMEC, 3))
        # BdotN_surf = np.sum(Bbs * surf.unitnormal(), axis=2)
        # BdotN = np.mean(np.abs(BdotN_surf))
        BdotN = np.max(np.abs((np.sum(Bbs * surf.unitnormal(), axis=2)) / np.linalg.norm(Bbs, axis=2)))
        # BdotNmax = np.max(np.abs(BdotN_surf))
        outstr = f"fun_coils#{info['Nfeval']} - J={J:.1e}, Jf={jf:.1e}, max⟨B·n⟩/B={BdotN:.1e}"  # , B·n max={BdotNmax:.1e}"
        # outstr += f", ║∇J coils║={np.linalg.norm(JF.dJ()):.1e}"
        # outstr += f", C-C-Sep={Jccdist.shortest_distance():.2f}"
        cl_string = ", ".join([f"{j.J():.1f}" for j in Jls])
        kap_string = ", ".join(f"{np.max(c.kappa()):.1f}" for c in base_curves)
        msc_string = ", ".join(f"{j.J():.1f}" for j in Jmscs)
        outstr += f" lengths=sum([{cl_string}])={sum(j.J() for j in Jls):.1f}, curv=[{kap_string}],msc=[{msc_string}]"
        print(outstr)
        # print(f"Currents: {[c.current.get_value() for c in coils]}")
        # print(f"dofs: {[c.curve.x for c in coils]}")
    return J, grad
##########################################################################################
##########################################################################################
## The function fun defined below is used to optimize the coils and the surface together.
def fun(dofs, prob_jacobian=None, info={'Nfeval': 0}):
    info['Nfeval'] += 1
    J = prob.objective() + coils_objective_weight * JF.J()
    if info['Nfeval'] > MAXFEV_single_stage+1 and J < JACOBIAN_THRESHOLD:
        return J, [0] * len(dofs)
    JF.x = dofs[:-number_vmec_dofs]
    prob.x = dofs[-number_vmec_dofs:]
    bs.set_points(surf.gamma().reshape((-1, 3)))
    os.chdir(vmec_results_path)
    J_stage_1 = prob.objective()
    J_stage_2 = coils_objective_weight * JF.J()
    J = J_stage_1 + J_stage_2
    if J > JACOBIAN_THRESHOLD or np.isnan(J):
        proc0_print(f"Exception caught during function evaluation with J={J}. Returning J={JACOBIAN_THRESHOLD}")
        J = JACOBIAN_THRESHOLD
        grad_with_respect_to_surface = [0] * number_vmec_dofs
        grad_with_respect_to_coils = [0] * len(JF.x)
    else:
        proc0_print(f"fun#{info['Nfeval']}: Objective function = {J}")
        prob_dJ = prob_jacobian.jac(prob.x)
        ## Finite differences for the second-stage objective function
        coils_dJ = JF.dJ()
        ## Mixed term - derivative of squared flux with respect to the surface shape
        n = surf.normal()
        absn = np.linalg.norm(n, axis=2)
        B = bs.B().reshape((nphi_VMEC, ntheta_VMEC, 3))
        dB_by_dX = bs.dB_by_dX().reshape((nphi_VMEC, ntheta_VMEC, 3, 3))
        Bcoil = bs.B().reshape(n.shape)
        unitn = n * (1./absn)[:, :, None]
        Bcoil_n = np.sum(Bcoil*unitn, axis=2)
        mod_Bcoil = np.linalg.norm(Bcoil, axis=2)
        B_n = Bcoil_n
        B_diff = Bcoil
        B_N = np.sum(Bcoil * n, axis=2)
        assert Jf.definition == "local"
        dJdx = (B_n/mod_Bcoil**2)[:, :, None] * (np.sum(dB_by_dX*(n-B*(B_N/mod_Bcoil**2)[:, :, None])[:, :, None, :], axis=3))
        dJdN = (B_n/mod_Bcoil**2)[:, :, None] * B_diff - 0.5 * (B_N**2/absn**3/mod_Bcoil**2)[:, :, None] * n
        deriv = surf.dnormal_by_dcoeff_vjp(dJdN/(nphi_VMEC*ntheta_VMEC)) + surf.dgamma_by_dcoeff_vjp(dJdx/(nphi_VMEC*ntheta_VMEC))
        mixed_dJ = Derivative({surf: deriv})(surf)
        ## Put both gradients together
        grad_with_respect_to_coils = coils_objective_weight * coils_dJ
        grad_with_respect_to_surface = np.ravel(prob_dJ) + coils_objective_weight * mixed_dJ
    grad = np.concatenate((grad_with_respect_to_coils, grad_with_respect_to_surface))
    return J, grad
##########################################################################################
#############################################################
## Perform optimization
#############################################################
##########################################################################################
max_mode_previous = 0
free_coil_dofs_all = JF.dofs_free_status
for iteration, max_mode in enumerate(max_mode_array):
    max_mode_previous+=1
    proc0_print(f'###############################################')
    proc0_print(f'  Performing optimization for max_mode={max_mode}')
    proc0_print(f'###############################################')
    vmec.indata.mpol = maxmodes_mpol_mapping[max_mode]
    vmec.indata.ntor = maxmodes_mpol_mapping[max_mode]
    MAXITER_single_stage = MAXITER_single_stage_mpol_mapping[max_mode]
    MAXFEV_single_stage = MAXFEV_single_stage_mpol_mapping[max_mode]
    quasisymmetry_weight = quasisymmetry_weight_mpol_mapping[max_mode]
    surf.fix_all()
    surf.fixed_range(mmin=0, mmax=max_mode, nmin=-max_mode, nmax=max_mode, fixed=False)
    surf.fix("rc(0,0)")
    number_vmec_dofs = int(len(surf.x))
    
    def aspect_ratio_max_objective(vmec): return np.max((vmec.aspect()-aspect_ratio_target,0))
    aspect_ratio_max_optimizable = make_optimizable(aspect_ratio_max_objective, vmec)
    objective_tuple = [(aspect_ratio_max_optimizable.J, 0, aspect_ratio_weight)]
    
    # def iota_min_objective(vmec): return np.min((np.mean(np.abs(vmec.wout.iotaf))-(iota_min_QA if QA_or_QH in ['QA','simple'] else iota_min_QH),0))
    def iota_min_objective(vmec): return np.min((np.min(np.abs(vmec.wout.iotaf))-(iota_min_QA if QA_or_QH in ['QA','simple'] else iota_min_QH),0))
    iota_min_optimizable = make_optimizable(iota_min_objective, vmec)
    objective_tuple.append((iota_min_optimizable.J, 0, weight_iota))
    # objective_tuple.append((vmec.mean_iota, iota_min_QA if QA_or_QH in ['QA','simple'] else iota_min_QH, weight_iota))

    # if QA_or_QH in ['QA', 'simple']:
    #     objective_tuple.append((vmec.mean_iota, iota_QA_simple, weight_iota))
    optMirror = make_optimizable(partial_MirrorRatioPen, vmec)
    proc0_print(f"Mirror before optimization: {optMirror.J()}")
    if QA_or_QH in ['QA', 'QH']:
        qs = QuasisymmetryRatioResidual(vmec, quasisymmetry_target_surfaces, helicity_m=1, helicity_n=-1 if QA_or_QH == 'QH' else 0)
        objective_tuple.append((qs.residuals, 0, quasisymmetry_weight))
    elif 'QI' in QA_or_QH:
        print('No QI functions')
        exit()
    else:
        objective_tuple.append((optMirror.J, 0, mirror_weight))
    prob = LeastSquaresProblem.from_tuples(objective_tuple)
    dofs = np.concatenate((JF.x, vmec.x))
    bs.set_points(surf.gamma().reshape((-1, 3)))
    Jf = SquaredFlux(surf, bs, definition="local")
    proc0_print(f"Aspect ratio before optimization: {vmec.aspect()}")
    proc0_print(f"Mean iota before optimization: {vmec.mean_iota()}")
    if QA_or_QH in ['QA', 'QH']: proc0_print(f"Quasisymmetry objective before optimization: {qs.total()}")
    proc0_print(f"Magnetic well before optimization: {vmec.vacuum_well()}")
    proc0_print(f"Squared flux before optimization: {Jf.J()}")
    
    proc0_print(f'  Performing stage 2 optimization with ~{MAXITER_stage_2} iterations')
    if comm_world.rank == 0:
        JF.full_unfix(free_coil_dofs_all)
        res = minimize(fun_coils, dofs[:-number_vmec_dofs], jac=True, args=({'Nfeval': 0}), method='L-BFGS-B', options={'maxiter': MAXITER_stage_2, 'maxcor': 300}, tol=1e-12)
        dofs[:-number_vmec_dofs] = res.x
        # if max_mode_previous==1:
        #     res = minimize(fun_coils, dofs[:-number_vmec_dofs]*1.02, jac=True, args=({'Nfeval': 0}), method='L-BFGS-B', options={'maxiter': MAXITER_stage_2, 'maxcor': 300}, tol=1e-12)
        #     dofs[:-number_vmec_dofs] = res.x
        #     res = minimize(fun_coils, dofs[:-number_vmec_dofs]*1.01, jac=True, args=({'Nfeval': 0}), method='L-BFGS-B', options={'maxiter': MAXITER_stage_2, 'maxcor': 300}, tol=1e-12)
        #     dofs[:-number_vmec_dofs] = res.x
    mpi.comm_world.Bcast(dofs, root=0)
    JF.x = dofs[:-number_vmec_dofs]
    bs.set_points(surf.gamma().reshape((-1, 3)))
    Bbs = bs.B().reshape((nphi_VMEC, ntheta_VMEC, 3))
    BdotN_surf = np.sum(Bbs * surf.unitnormal(), axis=2)
        
    if optimize_stage_1_with_coils:
        def JF_objective(vmec):
            bs.set_points(vmec.boundary.gamma().reshape((-1, 3)))
            return JF.J()
        JF_objective_optimizable = make_optimizable(JF_objective, vmec)
        Jf_residual = JF_objective_optimizable.J()
        prob_residual = prob.objective()
        new_Jf_weight = (prob_residual/Jf_residual)**2
        objective_tuples_with_coils = tuple(objective_tuple)+tuple([(JF_objective_optimizable.J, 0, new_Jf_weight)])
        prob_with_coils = LeastSquaresProblem.from_tuples(objective_tuples_with_coils)
        proc0_print(f'  Performing stage 1 optimization with coils with ~{MAXITER_stage_1} iterations')
        free_coil_dofs_all = JF.dofs_free_status
        JF.fix_all()
        least_squares_mpi_solve(prob_with_coils, mpi, grad=True, rel_step=1e-5, abs_step=1e-7, max_nfev=MAXITER_stage_1, ftol=1e-04, xtol=1e-04, gtol=1e-04)
        JF.full_unfix(free_coil_dofs_all)
        
    mpi.comm_world.Bcast(dofs, root=0)
    JF.x = dofs[:-number_vmec_dofs]
    bs.set_points(surf.gamma().reshape((-1, 3)))
    Bbs = bs.B().reshape((nphi_VMEC, ntheta_VMEC, 3))
    BdotN_surf = np.sum(Bbs * surf.unitnormal(), axis=2) / np.linalg.norm(Bbs, axis=2)

    if comm_world.rank == 0:
        curves_to_vtk(base_curves, os.path.join(coils_results_path, f"base_curves_after_stage2_maxmode{max_mode}"))
        curves_to_vtk(curves, os.path.join(coils_results_path, f"curves_after_stage2_maxmode{max_mode}"))
        pointData = {"B.n/B": BdotN_surf[:, :, None]}
        surf.to_vtk(os.path.join(coils_results_path, f"surf_after_stage2_maxmode{max_mode}"), extra_data=pointData)
    bs.set_points(surf_big.gamma().reshape((-1, 3)))
    Bbs = bs.B().reshape((nphi_big, ntheta_big, 3))
    BdotN_surf = np.sum(Bbs * surf_big.unitnormal(), axis=2) / np.linalg.norm(Bbs, axis=2)
    if comm_world.rank == 0:
        pointData = {"B.n/B": BdotN_surf[:, :, None]}
        surf_big.to_vtk(os.path.join(coils_results_path, f"surf_big_after_stage2_maxmode{max_mode}"), extra_data=pointData)
    bs.set_points(surf.gamma().reshape((-1, 3)))
    proc0_print(f'  Performing single stage optimization with ~{MAXITER_single_stage} iterations')
    x0 = np.copy(np.concatenate((JF.x, vmec.x)))
    dofs = np.concatenate((JF.x, vmec.x))
    with MPIFiniteDifference(prob.objective, mpi, diff_method=diff_method, abs_step=finite_difference_abs_step, rel_step=finite_difference_rel_step) as prob_jacobian:
        if mpi.proc0_world:
            res = minimize(fun, dofs, args=(prob_jacobian, {'Nfeval': 0}), jac=True, method='BFGS', options={'maxiter': MAXITER_single_stage, 'maxfev': MAXFEV_single_stage, 'gtol': ftol, 'ftol': ftol}, tol=ftol)
    mpi.comm_world.Bcast(dofs, root=0)
    Bbs = bs.B().reshape((nphi_VMEC, ntheta_VMEC, 3))
    BdotN_surf = np.sum(Bbs * surf.unitnormal(), axis=2) / np.linalg.norm(Bbs, axis=2)
    if comm_world.rank == 0:
        curves_to_vtk(base_curves, os.path.join(coils_results_path, f"base_curves_opt_maxmode{max_mode}"))
        curves_to_vtk(curves, os.path.join(coils_results_path, f"curves_opt_maxmode{max_mode}"))
        pointData = {"B.n/B": BdotN_surf[:, :, None]}
        surf.to_vtk(os.path.join(coils_results_path, f"surf_opt_maxmode{max_mode}"), extra_data=pointData)
    bs.set_points(surf_big.gamma().reshape((-1, 3)))
    Bbs = bs.B().reshape((nphi_big, ntheta_big, 3))
    BdotN_surf = np.sum(Bbs * surf_big.unitnormal(), axis=2) / np.linalg.norm(Bbs, axis=2)
    if comm_world.rank == 0:
        pointData = {"B.n/B": BdotN_surf[:, :, None]}
        surf_big.to_vtk(os.path.join(coils_results_path, f"surf_big_opt_maxmode{max_mode}"), extra_data=pointData)
    bs.set_points(surf.gamma().reshape((-1, 3)))
    Bbs = bs.B().reshape((nphi_VMEC, ntheta_VMEC, 3))
    bs.save(os.path.join(coils_results_path, f"biot_savart_maxmode{max_mode}.json"))
    vmec.write_input(os.path.join(this_path, f'input.maxmode{max_mode}'))
bs.save(os.path.join(coils_results_path, "biot_savart_opt.json"))
vmec.write_input(os.path.join(this_path, 'input.final'))
proc0_print(f"Mirror before optimization: {optMirror.J()}")
proc0_print(f"Aspect ratio after optimization: {vmec.aspect()}")
proc0_print(f"Mean iota after optimization: {vmec.mean_iota()}")
if QA_or_QH in ['QA', 'QH']: proc0_print(f"Quasisymmetry objective after optimization: {qs.total()}")
proc0_print(f"Magnetic well after optimization: {vmec.vacuum_well()}")
proc0_print(f"Squared flux after optimization: {Jf.J()}")
BdotN_surf = np.sum(Bbs * surf.unitnormal(), axis=2)
BdotN = np.mean(np.abs(BdotN_surf))
BdotNmax = np.max(np.abs(BdotN_surf))
outstr = f"Coil parameters: ⟨B·n⟩={BdotN:.1e}, B·n max={BdotNmax:.1e}"
outstr += f", ║∇J coils║={np.linalg.norm(JF.dJ()):.1e}"
# outstr += f", C-C-Sep={Jccdist.shortest_distance():.2f}"
cl_string = ", ".join([f"{j.J():.1f}" for j in Jls])
kap_string = ", ".join(f"{np.max(c.kappa()):.1f}" for c in base_curves)
msc_string = ", ".join(f"{j.J():.1f}" for j in Jmscs)
outstr += f" lengths=sum([{cl_string}])={sum(j.J() for j in Jls):.1f}, curv=[{kap_string}], msc=[{msc_string}]"
proc0_print(outstr)
