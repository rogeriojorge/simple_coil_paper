#!/usr/bin/env python3
import os
import shutil
import numpy as np
from simsopt import load
from pathlib import Path
from simsopt import make_optimizable
from simsopt._core.optimizable import Optimizable
from scipy.optimize import minimize, least_squares
from simsopt.util import MpiPartition, proc0_print, comm_world
from simsopt._core.derivative import Derivative, derivative_dec
from simsopt.mhd import Vmec, QuasisymmetryRatioResidual
from simsopt._core.finite_difference import MPIFiniteDifference
from simsopt.field import BiotSavart, Current, coils_via_symmetries
from simsopt.objectives import SquaredFlux, QuadraticPenalty, LeastSquaresProblem
from simsopt.geo import (CurveLength, CurveCurveDistance, MeanSquaredCurvature, SurfaceRZFourier, LinkingNumber,
                         CurveSurfaceDistance, LpCurveCurvature, ArclengthVariation, curves_to_vtk, create_equally_spaced_curves)
from simsopt.solve import least_squares_mpi_solve, least_squares_serial_solve
from scipy.optimize import least_squares

mpi = MpiPartition()
parent_path = str(Path(__file__).parent.resolve())
os.chdir(parent_path)
nfp = 3
###########
# IMPORTANT
# WHEN DOING coils_objective_weight = 0, ncoils = 0, order_coils = 1
# Not able to get same solutions as least_squares_mpi_solve only
## Other problems to look at
# do a 2 and 3 nfp QA, and a 4 and 5 nfp QH
# generalize this to allow doing stellarators with scans in iota/aspect ratio and different types of quasisymmetry/QI
## NOT ABLE TO SET TWO DIFFERENT PHIEDGES FOR TWO DIFFERENT VMEC OBJECTS
# Optimize for one configuration in vacuum and the other at finite beta with exactly the same coils, just changing the currents
##########################################################################################
############## Input parameters
##########################################################################################
optimize_stage_1 = False
optimize_stage_2 = True
optimize_stage_1_with_coils = False
optimize_single_stage = True
MAXITER_stage_1 = 8
MAXITER_stage_2 = 2500
MAXITER_single_stage = 50
MAXFEV_single_stage = 75
max_mode_array = [1]*5+[2]*4+[3]*4+[4]*4
start_from_tokamak = False
warm_start = True # True

ncoils = 6 #7 #7 #6 #4
order_coils = 6 #6 #5 # 4 #5
R1 = 0.49 #0.45 #0.47 # 0.69 # 0.67
LENGTH_THRESHOLD = 2.6 #2.4 #2.2 # 2.4 # 3.8
LENGTH_CON_WEIGHT = 0.0019 #0.0042 #0.0069 # 6.9e-3 # 5.1e-1
CURVATURE_THRESHOLD = 7.7 #1.9e1 #6.5 # 2.4e+1 # 6.5
CURVATURE_WEIGHT = 1.2e-5 #1.6e-6 #1.4e-5 # 1.2e-3 # 7.1e-3  # Weight for the curvature penalty in the objective function
MSC_THRESHOLD = 1.4e1 #1.5e1 #1.4e+1 # 6.8 # 5.6
MSC_WEIGHT = 6.5e-6 #1.7e-6 #3.1e-5 # 3.5e-6 # 2.4e-6  # Weight for the mean squared curvature penalty in the objective function
CC_THRESHOLD = 0.0806 #0.0841 #0.0758 # 0.089 # 0.11
CC_WEIGHT = 4.9 #3.0 #1.1e+3 # 7.7 # 2.7e+2

JACOBIAN_THRESHOLD = 20

maxmodes_mpol_mapping = {1: 5, 2: 5, 3: 5, 4: 5, 5: 6}
aspect_ratio_target = 10.0
CS_THRESHOLD = 0.03
CS_WEIGHT = 2e4
nphi_VMEC = 30
ntheta_VMEC = 30
coils_objective_weight = 1e4
aspect_ratio_weight = 6e-2
iota_weight = 1e2
qs_weight = 1.0e2
R0 = 1.0
diff_method = "forward"
iota_target = 0.31
iota_min_QA = 0.31
iota_min_QH = 0.71
finite_difference_abs_step = 1e-7
finite_difference_rel_step = 1e-4
tol_minimize_single_stage = 1e-3
numquadpoints = 100
# ARCLENGTH_WEIGHT = 1e-9  # Weight for the arclength variation penalty in the objective function
quasisymmetry_target_surfaces = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

if start_from_tokamak: filenames = [f'input.nfp{nfp}_circular_QH', f'input.nfp{nfp}_circular_QA']
else:
    # if warm_start: filenames = [f'input.nfp{nfp}_QH_warm', f'input.nfp{nfp}_QA_warm']
    if warm_start: filenames = [f'input.nfp{nfp}_QH_final', f'input.nfp{nfp}_QA_final']
    else:          filenames = [f'input.nfp{nfp}_QH', f'input.nfp{nfp}_QA']
vmec_input_folder = 'vmec_inputs'
vmec_input_filenames = [os.path.join(parent_path, vmec_input_folder, filename) for filename in filenames]
##########################################################################################
##########################################################################################
directory = 'optimization'
directory+= f'_ncoils_{ncoils}_order_{order_coils}_R1_{R1}_length_target_{LENGTH_THRESHOLD}_weight_{coils_objective_weight}_max_curvature_{CURVATURE_THRESHOLD}_weight_{CURVATURE_WEIGHT}_msc_{MSC_THRESHOLD}_weight_{MSC_WEIGHT}_cc_{CC_THRESHOLD}_weight_{CC_WEIGHT}_QH_{CS_THRESHOLD}_weight_{CS_WEIGHT}'
vmec_verbose = False
# Create output directories
this_path = os.path.join(parent_path, directory)
os.makedirs(this_path, exist_ok=True)
os.chdir(this_path)
shutil.copyfile(os.path.join(parent_path, 'single_stage_optimization.py'), os.path.join(this_path, 'single_stage_optimization.py'))
vmec_results_path = os.path.join(this_path, "vmec")
coils_results_path = os.path.join(this_path, "coils")
if comm_world.rank == 0:
    os.makedirs(vmec_results_path, exist_ok=True)
    os.makedirs(coils_results_path, exist_ok=True)
##########################################################################################
##########################################################################################
# Stage 1
proc0_print(f' Using vmec input files {filenames}:')
vmecs = [Vmec(filename, mpi=mpi, verbose=vmec_verbose, nphi=nphi_VMEC, ntheta=ntheta_VMEC, range_surface='half period') for filename in vmec_input_filenames]
## NOT ABLE TO SET TWO DIFFERENT PHIEDGES FOR TWO DIFFERENT VMEC OBJECTS
# for i in range(len(vmecs)):
#     vmecs[i].run()
#     vmecs[i].indata.phiedge = vmecs[i].indata.phiedge/vmecs[i].wout.volavgB
#     vmecs[i].run()
#     # print(vmec_input_filenames[i])
#     # print(vmecs[i].indata.phiedge)
#     # print(vmecs[i].wout.volavgB)
#     # print(vmecs[i].x)
# print(vmec_input_filenames)
# print(vmecs[0].indata.phiedge)
# print(vmecs[1].indata.phiedge)
# vmecs[0].run()
# vmecs[1].run()
# print(vmecs[0].wout.volavgB)
# print(vmecs[1].wout.volavgB)

# exit()
surfs = [vmec.boundary for vmec in vmecs]
nphi_big   = nphi_VMEC * 2 * nfp + 1
ntheta_big = ntheta_VMEC + 1
quadpoints_theta = np.linspace(0, 1, ntheta_big)
quadpoints_phi   = np.linspace(0, 1, nphi_big)
surfs_big = [SurfaceRZFourier(dofs=surf.dofs, nfp=nfp, mpol=surf.mpol, ntor=surf.ntor, quadpoints_phi=quadpoints_phi, quadpoints_theta=quadpoints_theta) for surf in surfs]
##########################################################################################
##########################################################################################
#Stage 2
def process_surface_and_flux(bs, surf, surf_big=None, new_OUT_DIR=coils_results_path, prefix="", plot=True):
    bs.set_points(surf.gamma().reshape((-1, 3)))
    Bbs = bs.B().reshape((nphi_VMEC, ntheta_VMEC, 3))
    BdotN = (np.sum(Bbs * surf.unitnormal(), axis=2)) / np.linalg.norm(Bbs, axis=2)
    # maxBdotN = np.max(np.abs(BdotN))
    if plot:
        pointData = {"B.n/B": BdotN[:, :, None]}
        surf.to_vtk(os.path.join(new_OUT_DIR, f"surf_{prefix}_halfnfp"), extra_data=pointData)
        if surf_big is not None:
            bs.set_points(surf_big.gamma().reshape((-1, 3)))
            Bbs = bs.B().reshape((nphi_big, ntheta_big, 3))
            BdotN = (np.sum(Bbs * surf_big.unitnormal(), axis=2)) / np.linalg.norm(Bbs, axis=2)
            pointData = {"B.n/B": BdotN[:, :, None]}
            surf_big.to_vtk(os.path.join(new_OUT_DIR, f"surf_{prefix}_big"), extra_data=pointData)
    bs.set_points(surf.gamma().reshape((-1, 3)))
    Jf = SquaredFlux(surf, bs, definition="local")
    return Jf
# if warm_start:
#     if ncoils==4:
#         filename_coils_1 = f"ncoils_{ncoils}_order_{order_coils}_R1_0.41_length_target_{LENGTH_THRESHOLD}_weight_0.0014_max_curvature_9.4_weight_0.00077_msc_2.5e+01_weight_0.00018_cc_0.1_weight_1.2e+01_QH.json"
#         filename_coils_2 = f"ncoils_{ncoils}_order_{order_coils}_R1_0.41_length_target_{LENGTH_THRESHOLD}_weight_0.0014_max_curvature_9.4_weight_0.00077_msc_2.5e+01_weight_0.00018_cc_0.1_weight_1.2e+01_QA.json"
#     elif ncoils==5:
#         filename_coils_1 = f"ncoils_{ncoils}_order_{order_coils}_R1_0.49_length_target_3.6_weight_0.0072_max_curvature_2.4e+01_weight_0.0099_msc_9.0_weight_4.6e-06_cc_0.1_weight_3.4e+02_QH.json"
#         filename_coils_2 = f"ncoils_{ncoils}_order_{order_coils}_R1_0.49_length_target_3.6_weight_0.0072_max_curvature_2.4e+01_weight_0.0099_msc_9.0_weight_4.6e-06_cc_0.1_weight_3.4e+02_QA.json"
#     else:
#         raise ValueError('ncoils must be 4 or 5')
#     bs1 = load(os.path.join(parent_path, vmec_input_folder, filename_coils_1))
#     bs2 = load(os.path.join(parent_path, vmec_input_folder, filename_coils_2))
#     bss = [bs1, bs2]
#     coils = [bs.coils for bs in bss]
#     base_curves = [bs1.coils[i].curve for i in range(ncoils)]
#     base_currents = [[bs.coils[i].current for i in range(ncoils)] for bs in bss]
# else:
base_curves = create_equally_spaced_curves(ncoils, nfp, stellsym=True, R0=R0, R1=R1, order=order_coils, numquadpoints=numquadpoints)
base_currents = [[Current(1.0) * (1e5) for i in range(ncoils)] for j in range(len(surfs))]
base_currents[0][0].fix_all()
coils = [coils_via_symmetries(base_curves, base_currents[j], nfp, True) for j in range(len(surfs))]
bss = [BiotSavart(coil) for coil in coils]
curves = [c.curve for c in coils[0]]
if comm_world.rank == 0:
    curves_to_vtk(curves, os.path.join(coils_results_path, "curves_init"))
    curves_to_vtk(base_curves, os.path.join(coils_results_path, "curves_init_halfnfp"))
# Jf1, _ = process_surface_and_flux(bs1, surf1, surf_big=surf_big1, new_OUT_DIR=coils_results_path, prefix="surf1_init")
# Jf2, _ = process_surface_and_flux(bs2, surf2, surf_big=surf_big2, new_OUT_DIR=coils_results_path, prefix="surf2_init")

Jfs = [process_surface_and_flux(bs, surf, surf_big=surf_big, new_OUT_DIR=coils_results_path, prefix=f"surf{i}_init") for i, (bs, surf, surf_big) in enumerate(zip(bss, surfs, surfs_big))]
Jf = sum(Jfs)
##########################################################################################
##########################################################################################
Jls = [CurveLength(c) for c in base_curves]
Jccdist = CurveCurveDistance(curves, CC_THRESHOLD, num_basecurves=len(curves))
Jcs = [LpCurveCurvature(c, 2, CURVATURE_THRESHOLD) for i, c in enumerate(base_curves)]
Jmscs = [MeanSquaredCurvature(c) for c in base_curves]
# Jals = [ArclengthVariation(c) for c in base_curves]
# J_LENGTH = LENGTH_WEIGHT * sum(Jls)
J_CC = CC_WEIGHT * Jccdist
J_CURVATURE = CURVATURE_WEIGHT * sum(Jcs)
J_MSC = MSC_WEIGHT * sum(QuadraticPenalty(J, MSC_THRESHOLD, "max") for J in Jmscs)
# J_ALS = ARCLENGTH_WEIGHT * sum(Jals)
Jcsdist = CS_WEIGHT * sum([CurveSurfaceDistance(base_curves, surf, CS_THRESHOLD) for surf in surfs])
J_LENGTH_PENALTY = LENGTH_CON_WEIGHT * sum(QuadraticPenalty(J, LENGTH_THRESHOLD, "max") for J in Jls)
JF = Jf + J_CC + J_LENGTH_PENALTY + J_CURVATURE + J_MSC + LinkingNumber(curves, 2) + Jcsdist
##########################################################################################
##########################################################################################
def save_results(surf, vmec, Jf, bs, filename, output_name, qs, save_coils=True):
    QA_or_QH = 'QH' if 'QH' in filename else 'QA'
    proc0_print('  Saving', QA_or_QH)
    if save_coils:
        curves_to_vtk(curves, os.path.join(coils_results_path, f"curves_{QA_or_QH}_{output_name}"))
        curves_to_vtk(base_curves, os.path.join(coils_results_path, f"base_curves_{QA_or_QH}_{output_name}"))
    bs.save(os.path.join(coils_results_path, f"biot_savart_{QA_or_QH}_{output_name}.json"))
    vmec.write_input(os.path.join(this_path, f'input.{QA_or_QH}_{output_name}'))
    proc0_print(f"Aspect ratio after optimization: {vmec.aspect()}")
    proc0_print(f"Mean iota after optimization: {vmec.mean_iota()}")
    proc0_print(f"Quasisymmetry objective after optimization: {qs.total()}")
    proc0_print(f"Squared flux after optimization: {Jf.J()}")
    proc0_print(f"volavgB after optimization: {vmec.wout.volavgB}")
    Bbs = bs.B().reshape((nphi_VMEC, ntheta_VMEC, 3))
    BdotN_surf = (np.sum(Bbs * surf.unitnormal(), axis=2)) / np.linalg.norm(Bbs, axis=2)
    BdotN = np.mean(np.abs(BdotN_surf))
    BdotNmax = np.max(np.abs(BdotN_surf))
    outstr = f"Coil parameters: ⟨B·n⟩={BdotN:.1e}, B·n max={BdotNmax:.1e}"
    outstr += f", ║∇J coils║={np.linalg.norm(JF.dJ()):.1e}, C-C-Sep={Jccdist.shortest_distance():.2f}"
    cl_string = ", ".join([f"{j.J():.1f}" for j in Jls])
    kap_string = ", ".join(f"{np.max(c.kappa()):.1f}" for c in base_curves)
    msc_string = ", ".join(f"{j.J():.1f}" for j in Jmscs)
    outstr += f" lengths=sum([{cl_string}])={sum(j.J() for j in Jls):.1f}, curv=[{kap_string}], msc=[{msc_string}]"
    proc0_print(outstr)
    ##########################################################################################
##########################################################################################
proc0_print('  Starting optimization')
##########################################################################################
# Initial stage 2 optimization
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
        BdotNs = [np.max(np.abs((np.sum(bs.B().reshape((nphi_VMEC, ntheta_VMEC, 3)) * surf.unitnormal(), axis=2)) / np.linalg.norm(bs.B().reshape((nphi_VMEC, ntheta_VMEC, 3)), axis=2))) for (bs, surf) in zip(bss, surfs)]
        BdotN = max(BdotNs)
        outstr = f"fun_coils#{info['Nfeval']} - J={J:.1e}, Jf={jf:.1e}, max⟨B·n⟩/B={BdotN:.1e}"
        outstr += f", ║∇J coils║={np.linalg.norm(JF.dJ()):.1e}, C-C-Sep={Jccdist.shortest_distance():.2f}"
        cl_string = ", ".join([f"{j.J():.1f}" for j in Jls])
        kap_string = ", ".join(f"{np.max(c.kappa()):.1f}" for c in base_curves)
        msc_string = ", ".join(f"{j.J():.1f}" for j in Jmscs)
        outstr += f" lengths=sum([{cl_string}])={sum(j.J() for j in Jls):.1f}, curv=[{kap_string}], msc=[{msc_string}]"
        print(outstr)
    return J, grad
##########################################################################################
##########################################################################################
## The function fun defined below is used to optimize the coils and the surface together.
def compute_jacobian(dofs, prob_jacobian=None, info={'Nfeval': 0}):
    if info['Nfeval'] > MAXFEV_single_stage:
        return [0] * len(dofs)
    JF.x = dofs[:-number_vmec_dofs]
    prob.x = dofs[-number_vmec_dofs:]
    if prob_jacobian is None:
        grad_with_respect_to_surface, grad_with_respect_to_coils = [0] * number_vmec_dofs, [0] * len(JF.x)
    else:
        prob_dJ = prob_jacobian.jac(prob.x)
        coils_dJ = JF.dJ()
        def calculate_mixed_derivative(bs, surf):
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
            dJdx = (B_n/mod_Bcoil**2)[:, :, None] * (np.sum(dB_by_dX*(n-B*(B_N/mod_Bcoil**2)[:, :, None])[:, :, None, :], axis=3))
            dJdN = (B_n/mod_Bcoil**2)[:, :, None] * B_diff - 0.5 * (B_N**2/absn**3/mod_Bcoil**2)[:, :, None] * n
            deriv = surf.dnormal_by_dcoeff_vjp(dJdN/(nphi_VMEC*ntheta_VMEC)) + surf.dgamma_by_dcoeff_vjp(dJdx/(nphi_VMEC*ntheta_VMEC))
            return Derivative({surf: deriv})(surf)
        mixed_dJ = np.concatenate([calculate_mixed_derivative(bs, surf) for bs, surf in zip(bss, surfs)])
        grad_with_respect_to_coils = coils_objective_weight * coils_dJ
        grad_with_respect_to_surface = np.ravel(prob_dJ) + coils_objective_weight * mixed_dJ
    grad = np.concatenate((grad_with_respect_to_coils, grad_with_respect_to_surface))
    return grad
##########################################################################################
def fun(dofs, prob_jacobian=None, info={'Nfeval': 0}):
    info['Nfeval'] += 1
    JF.x = dofs[:-number_vmec_dofs]
    prob.x = dofs[-number_vmec_dofs:]
    [bs.set_points(surf.gamma().reshape((-1, 3))) for bs, surf in zip(bss, surfs)]
    J_stage_1 = prob.objective()
    J_stage_2 = coils_objective_weight * JF.J()
    J = J_stage_1 + J_stage_2
    if info['Nfeval'] > MAXFEV_single_stage and J < JACOBIAN_THRESHOLD:
        return J
    if J > JACOBIAN_THRESHOLD or np.isnan(J):
        proc0_print(f"Exception caught during function evaluation with J={J}. Returning J={JACOBIAN_THRESHOLD}")
        J = JACOBIAN_THRESHOLD
    else:
        proc0_print(f"fun#{info['Nfeval']}: Objective function = {J:.4f}")
    return J
##########################################################################################
class fun_surf_coils(Optimizable):
    def __init__(self, vmecs, JF):
        self.vmecs = vmecs
        self.JF = JF
        super().__init__(depends_on=[JF]+[vmec for vmec in vmecs])

    def J(self):
        """
        This returns the value of the quantity.
        """
        dofs = np.concatenate((JF.x, np.concatenate([vmec.x for vmec in vmecs])))
        return fun(dofs)
    
    def residuals(self):
        dofs = np.concatenate((JF.x, np.concatenate([vmec.x for vmec in vmecs])))
        return fun(dofs)

    def objective(self):
        dofs = np.concatenate((JF.x, np.concatenate([vmec.x for vmec in vmecs])))
        return fun(dofs)

    # @derivative_dec
    def dJ(self):
        """
        This returns the derivative of the quantity with respect to the curve dofs.
        """
        dofs = np.concatenate((JF.x, np.concatenate([vmec.x for vmec in vmecs])))
        # still needs to implement the derivative of the objective with respect to the coils
        return compute_jacobian(dofs, prob_jacobian)
    return_fn_map = {'J': J, 'dJ': dJ}

##########################################################################################
#############################################################
## Perform optimization
#############################################################
##########################################################################################
os.chdir(vmec_results_path)
max_mode_previous = 0
for iteration, max_mode in enumerate(max_mode_array):
    qss = []
    number_vmec_dofss = []
    objective_tuples  = ()
    proc0_print(f'############# Starting optimization for max_mode={max_mode} iteration {iteration+1} #################')
    for i, (surf, vmec, Jf, bs, filename) in enumerate(zip(surfs, vmecs, Jfs, bss, filenames)):
        surf.fix_all()
        surf.fixed_range(mmin=0, mmax=max_mode, nmin=-max_mode, nmax=max_mode, fixed=False)
        # surf.fix("rc(0,0)")
        number_vmec_dofss.append(int(len(surf.x)))
        QA_or_QH = 'QH' if 'QH' in filename else 'QA'
        qs = QuasisymmetryRatioResidual(vmec, quasisymmetry_target_surfaces, helicity_m=1, helicity_n=-1 if 'QH' in QA_or_QH else 0)
        qss.append(qs)
        proc0_print(f'  Looking at: {QA_or_QH}')
        if optimize_stage_1:
            proc0_print(f'    Stage 1 optimization (surface {i+1}/{len(surfs)})')
        proc0_print(f"    Aspect ratio before optimization: {vmec.aspect()}")
        proc0_print(f"    Mean iota before optimization: {vmec.mean_iota()}")
        # vmec.indata.phiedge = vmec.indata.phiedge/vmec.wout.volavgB;vmec.run()
        proc0_print(f"    Quasisymmetry objective before optimization: {qs.total()}")
        proc0_print(f"    Squared flux before optimization: {Jf.J()}")
        proc0_print(f"    volavgB before optimization: {vmec.wout.volavgB}")
        this_objective_tuple = [(vmec.aspect, aspect_ratio_target, aspect_ratio_weight), (qs.residuals, 0, qs_weight)]
        def iota_min_objective(vmec): return np.min((np.min(np.abs(vmec.wout.iotaf))-(iota_min_QA if QA_or_QH=='QA' else iota_min_QH),0))
        iota_min_optimizable = make_optimizable(iota_min_objective, vmec)
        this_objective_tuple.append((iota_min_optimizable.J, 0, iota_weight))
        objective_tuples+=tuple(this_objective_tuple)
        vmec.indata.mpol = maxmodes_mpol_mapping[max_mode]
        vmec.indata.ntor = maxmodes_mpol_mapping[max_mode]
        if optimize_stage_1 and max_mode_previous==0:
            prob = LeastSquaresProblem.from_tuples(this_objective_tuple)
            least_squares_mpi_solve(prob, mpi, grad=True, rel_step=1e-5, abs_step=1e-7, max_nfev=MAXITER_stage_1)
            if comm_world.rank == 0: save_results(surf, vmec, Jf, bs, filename, f'maxmode{max_mode}_{QA_or_QH}_after_stage1', qs, save_coils=False)
    number_vmec_dofs = np.sum(number_vmec_dofss)
    prob = LeastSquaresProblem.from_tuples(objective_tuples)
    if optimize_stage_1_with_coils and iteration>0:
        def JF_objective(vmecs):
            [bs.set_points(vmec.boundary.gamma().reshape((-1, 3))) for bs, vmec in zip(bss, vmecs)]
            return JF.J()
        JF_objective_optimizable = make_optimizable(JF_objective, vmecs)
        Jf_residual = JF_objective_optimizable.J()
        prob_residual = prob.objective()
        new_Jf_weight = 3e2*prob_residual/Jf_residual
        objective_tuples_with_coils = objective_tuples+tuple([(JF_objective_optimizable.J, 0, new_Jf_weight)])
        prob_with_coils = LeastSquaresProblem.from_tuples(objective_tuples_with_coils)
        proc0_print(f'  Performing stage 1 optimization with coils with ~{MAXITER_stage_1} iterations')
        
        free_coil_dofs_all = JF.dofs_free_status
        # JF.fix_all()
        
        # Varying currents in least_squares_mpi_solve
        currents_here = np.ravel([Jf.x[0:ncoils] for Jf in Jfs])
        free_coil_dofs_currents = np.isin(JF.x, currents_here)
        JF.fix_all()
        JF.full_unfix(free_coil_dofs_currents)
        
        least_squares_mpi_solve(prob_with_coils, mpi, grad=True, rel_step=1e-5, abs_step=1e-7, max_nfev=8 if iteration==0 else MAXITER_stage_1)
        JF.full_unfix(free_coil_dofs_all)
        if comm_world.rank == 0:
            for (surf, vmec, Jf, bs, filename, qs) in zip(surfs, vmecs, Jfs, bss, filenames, qss):
                save_results(surf, vmec, Jf, bs, filename, f'maxmode{max_mode}_{QA_or_QH}_after_stage1_with_coils', qs, save_coils=False)
        Jfs = [process_surface_and_flux(bs, surf, surf_big=surf_big, new_OUT_DIR=coils_results_path, prefix=f"{i}_maxmode{max_mode}_after_stage1_with_coils") for i, (bs, surf, surf_big) in enumerate(zip(bss, surfs, surfs_big))]
        Jf = sum(Jfs)
    dofs = np.concatenate((JF.x, np.concatenate([vmec.x for vmec in vmecs])))
    proc0_print(f'  len(dofs)={len(dofs)}')
    Jfs = [process_surface_and_flux(bs, surf, plot=False) for (bs, surf) in zip(bss, surfs)]
    Jf = sum(Jfs)
    if optimize_stage_2:
        proc0_print(f'  Performing stage 2 optimization with ~{MAXITER_stage_2} iterations')
        if mpi.proc0_world:
            res = minimize(fun_coils, dofs[:-number_vmec_dofs], jac=True, args=({'Nfeval': 0}), method='L-BFGS-B', options={'maxiter': MAXITER_stage_2, 'maxcor': 300}, tol=1e-12)
            print(res.message)
        Jfs = [process_surface_and_flux(bs, surf, surf_big=surf_big, new_OUT_DIR=coils_results_path, prefix=f"surf{i}_maxmode{max_mode}_after_stage_2") for i, (bs, surf, surf_big) in enumerate(zip(bss, surfs, surfs_big))]
        Jf = sum(Jfs)
        if comm_world.rank == 0:
            for (surf, vmec, Jf, bs, filename, qs) in zip(surfs, vmecs, Jfs, bss, filenames, qss):
                save_results(surf, vmec, Jf, bs, filename, f'maxmode{max_mode}_after_stage2', qs)
    proc0_print(f'  Performing single stage optimization with ~{MAXITER_single_stage} iterations')
    x0 = np.copy(np.concatenate((JF.x, np.concatenate([vmec.x for vmec in vmecs]))))
    dofs = np.concatenate((JF.x, np.concatenate([vmec.x for vmec in vmecs])))
    if optimize_single_stage:
        with MPIFiniteDifference(prob.objective, mpi, diff_method=diff_method, abs_step=finite_difference_abs_step, rel_step=finite_difference_rel_step) as prob_jacobian:
            if mpi.proc0_world:
                prob_single_stage = fun_surf_coils(vmecs, JF)
                res = minimize(fun, dofs, args=(prob_jacobian, {'Nfeval': 0}), jac=compute_jacobian, method='BFGS', options={'maxiter': MAXITER_single_stage}, tol=tol_minimize_single_stage)
                print(res.message)
                # least_squares_serial_solve(prob_single_stage, mpi, rel_step=1e-5, abs_step=1e-7, max_nfev=MAXITER_single_stage)
                # res = least_squares(fun, dofs, args=(prob_jacobian, {'Nfeval': 0}), jac=compute_jacobian, ftol=tol_minimize_single_stage, gtol=tol_minimize_single_stage, xtol=tol_minimize_single_stage, max_nfev=MAXITER_single_stage, verbose=2)
                dofs = res.x
    mpi.comm_world.Bcast(dofs, root=0)
    Jfs = [process_surface_and_flux(bs, surf, surf_big=surf_big, new_OUT_DIR=coils_results_path, prefix=f"surf{i}_maxmode{max_mode}") for i, (bs, surf, surf_big) in enumerate(zip(bss, surfs, surfs_big))]
    Jf = sum(Jfs)
    JF.x = dofs[:-number_vmec_dofs]
    max_mode_previous = max_mode
    for i, (surf, vmec, Jf, bs, filename, qs) in enumerate(zip(surfs, vmecs, Jfs, bss, filenames, qss)):
        save_results(surf, vmec, Jf, bs, filename, f'surf{i}_maxmode{max_mode}_after_singlestage', qs)

if comm_world.rank == 0:
    for i, (surf, vmec, Jf, bs, filename, qs) in enumerate(zip(surfs, vmecs, Jfs, bss, filenames, qss)):
        save_results(surf, vmec, Jf, bs, filename, 'final', qs)
        [process_surface_and_flux(bs, surf, surf_big=surf_big, new_OUT_DIR=coils_results_path, prefix=f"{i}_opt") for i, (bs, surf, surf_big) in enumerate(zip(bss, surfs, surfs_big))]