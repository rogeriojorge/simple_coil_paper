#!/usr/bin/env python3

import os
import numpy as np
from simsopt.configs import get_ncsx_data
from simsopt.field import BiotSavart, coils_via_symmetries, Current, Coil
from simsopt.geo import QfmResidual, QfmSurface, SurfaceRZFourier, ToroidalFlux, Area, Volume, curves_to_vtk
from simsopt import load
from simsopt.mhd import Vmec
this_dir = os.path.dirname(os.path.abspath(__file__))

curves=load('/Users/rogeriojorge/local/microstability_optimization/src/joaquim_circular_coil/circurves_opt.json')

# coils_symmetries = coils_via_symmetries([curves[0],curves[1]], [Current(1) * 1e5, Current(1) * 1e5], 3, stellsym=False)
# curves_symmetries = [c.curve for c in coils_symmetries]
# curves_to_vtk(curves_symmetries, "curves_symmetries")
# bs = BiotSavart(coils_symmetries)
# mpol = 5
# ntor = 5
# stellsym = True
# nfp = 3
# vol_target=0.05
# constraint_weight = 1e-0
# nphi   = 32
# ntheta = 32
# extend_distance = -0.00
# tol = 1e-13
# maxiter = 1000
# n_iterations=1
# phis = np.linspace(0, 1/3, nphi, endpoint=True)

curves_to_vtk(curves, "curves")
currents = [Current(1) * 1e5, Current(1) * 1e5, Current(1) * 1e5, Current(1) * 1e5, Current(1) * 1e5, Current(1) * 1e5]
coils = [Coil(curv, curr) for (curv, curr) in zip(curves, currents)]
bs = BiotSavart(coils)
mpol = 5
ntor = 31
stellsym = True
nfp = 1
vol_target=0.065
# area_target=1.4967300929226603
constraint_weight = 1e-0
nphi   = 128
ntheta = 32
extend_distance = -0.00
phis = np.linspace(0, 1, nphi, endpoint=True)
tol = 1e-11
maxiter = 400
n_iterations = 5

do_SLSQP = False

nfp_factor = 3 if nfp==1 else 1

thetas = np.linspace(0, 1, ntheta, endpoint=True)
s = SurfaceRZFourier(mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
# # s.set_rc(0, 0*nfp_factor, 0.3782)
# s.set_rc(0, 0*nfp_factor, 0.400)
# s.set_rc(0, 1*nfp_factor, 0.0435)
# s.set_rc(0, 2*nfp_factor,-0.0145)
# s.set_rc(1,-2*nfp_factor,-0.0136)
# s.set_rc(1,-1*nfp_factor, 0.0032)
# s.set_rc(1, 0*nfp_factor, 0.0916)
# s.set_rc(1, 2*nfp_factor,-0.0141)
# # s.set_rc(2,-2*nfp_factor,-0.0036)
# # s.set_rc(2,-1*nfp_factor, 0.0049)
# # s.set_rc(2, 1*nfp_factor,-0.0065)
# # s.set_rc(2, 2*nfp_factor,-0.0019)
# s.set_zs(0, 1*nfp_factor, 0.0118)
# s.set_zs(0, 2*nfp_factor, 0.0017)
# s.set_zs(1,-2*nfp_factor,-0.0195)
# s.set_zs(1, 0*nfp_factor, 0.0900)
# # s.set_zs(1, 2*nfp_factor,-0.0145)
# # s.set_zs(2,-2*nfp_factor,-0.0076)
# # s.set_zs(2,-1*nfp_factor, 0.0058)
# # s.set_zs(2, 1*nfp_factor,-0.0120)
# # s.set_zs(2, 2*nfp_factor,-0.0044)

# s.set_rc(0, 0, 0.3975)
# s.set_rc(0, 1, -0.0024)
# s.set_rc(0, 2, -0.0028)
# s.set_rc(0, 1*nfp_factor, 0.0457)
# s.set_zs(0, 1*nfp_factor, 0.0194)
# s.set_rc(0, 4, -0.0013)
# s.set_rc(0, 5, -0.0009)
# s.set_rc(0, 2*nfp_factor,-0.021)
# s.set_rc(0, 5*nfp_factor,-0.0013)
# s.set_rc(1,-5*nfp_factor,-0.0011)
# s.set_rc(1,-4*nfp_factor, 0.0012)
# s.set_zs(1,-4*nfp_factor, 0.0015)
# s.set_rc(1,-3*nfp_factor, 0.0020)
# s.set_rc(1,-2*nfp_factor,-0.0199)
# s.set_zs(1,-2*nfp_factor,-0.0185)
# s.set_rc(1,-1*nfp_factor, 0.0022)
# s.set_rc(1,-1,-0.0010)
# s.set_rc(1, 0, 0.0862)
# s.set_zs(1, 0, 0.0808)
# s.set_rc(1, 1*nfp_factor,-0.0124)
# s.set_rc(1, 2*nfp_factor,-0.0147)
# s.set_zs(1, 2*nfp_factor,-0.0069)
# s.set_rc(1, 3*nfp_factor, 0.0016)
# s.set_zs(2,-4*nfp_factor, 0.0015)
# s.set_rc(2,-2*nfp_factor,-0.0023)
# s.set_zs(2,-2*nfp_factor,-0.0025)
# s.set_rc(2,-1*nfp_factor, 0.0039)
# s.set_zs(2,-1*nfp_factor, 0.0046)
# s.set_rc(2,-2,-0.0013)
# s.set_zs(2,-2,-0.0013)
# s.set_rc(2,-1,-0.0009)
# s.set_rc(2, 0,-0.0021)
# s.set_zs(2, 0,-0.0030)
# s.set_rc(2, 1,-0.0011)
# s.set_zs(2, 1,-0.0011)
# s.set_rc(2, 2,-0.001)
# s.set_zs(2, 2,-0.001)
# s.set_rc(2, 1*nfp_factor,-0.0017)
# s.set_zs(2, 1*nfp_factor,-0.0016)
# s.set_rc(2, 2*nfp_factor,-0.0019)
# s.set_zs(2, 2*nfp_factor,-0.0023)
# s.set_zs(2, 3*nfp_factor, 0.0011)
# s.set_rc(3,-4*nfp_factor, 0.0010)
# s.set_zs(3,-4*nfp_factor, 0.0010)
# s.set_rc(3,-2*nfp_factor,-0.0022)
# s.set_zs(3,-2*nfp_factor,-0.0022)
# s.set_rc(3, 2*nfp_factor,-0.0027)
# s.set_zs(3, 2*nfp_factor, 0.0029)

s.set_rc(0, 0, 0.3974)
s.set_rc(0, 1, -0.0026)
s.set_rc(0, 2, -0.0029)
s.set_rc(0, 1*nfp_factor, 0.0462)
s.set_zs(0, 1*nfp_factor, 0.0196)
s.set_rc(0, 4, -0.0018)
s.set_rc(0, 5, -0.001)
s.set_rc(0, 2*nfp_factor, -0.0212)
s.set_rc(0, 9, 0.0024)
s.set_zs(0, 9, -0.0009)
s.set_rc(0, 15, -0.0015)
s.set_rc(1, -5*nfp_factor, -0.0011)
s.set_rc(1, -4*nfp_factor, 0.0012)
s.set_zs(1, -4*nfp_factor, 0.0015)
s.set_rc(1, -3*nfp_factor, 0.002)
s.set_rc(1, -2*nfp_factor, -0.0198)
s.set_zs(1, -2*nfp_factor, -0.0181)
s.set_rc(1, -1*nfp_factor, 0.0055)
s.set_zs(1, -1*nfp_factor, 0.0019)
s.set_rc(1, -1, -0.0013)
s.set_rc(1, 0, 0.0856)
s.set_zs(1, 0, 0.0802)
s.set_rc(1, 1*nfp_factor, -0.007)
s.set_zs(1, 1*nfp_factor, 0.0008)
s.set_rc(1, 2*nfp_factor, -0.0131)
s.set_zs(1, 2*nfp_factor, -0.0084)
s.set_rc(1, 3*nfp_factor, 0.0023)
s.set_rc(2, -2*nfp_factor, -0.0021)
s.set_zs(2, -2*nfp_factor, -0.0025)
s.set_rc(2, -1*nfp_factor, 0.0034)
s.set_zs(2, -1*nfp_factor, 0.0026)
s.set_rc(2, -1*nfp_factor, 0.0047)
s.set_rc(2, 0, -0.0021)
s.set_zs(2, 0, -0.0028)
s.set_rc(2, 1, -0.0012)
s.set_zs(2, 1, -0.0011)
s.set_rc(2, 1*nfp_factor, -0.0019)
s.set_zs(2, 1*nfp_factor, -0.0016)
s.set_rc(2, 2*nfp_factor, -0.0019)
s.set_zs(2, 2*nfp_factor, -0.0023)
s.set_rc(2, 3*nfp_factor, 0.0007)
s.set_zs(2, 3*nfp_factor, 0.0011)
s.set_rc(3, -6*nfp_factor, -0.0021)
s.set_zs(3, -6*nfp_factor, -0.0022)
s.set_rc(3, 1*nfp_factor, -0.0039)
s.set_rc(3, 2*nfp_factor, -0.0027)
s.set_zs(3, 2*nfp_factor, 0.003)



# s=load('/Users/rogeriojorge/local/microstability_optimization/src/joaquim_circular_coil/qfmsurf_opt.json')

# mpol_boundary = 2
# ntor_boundary = 2
# phis = np.linspace(0, 1/6, nphi, endpoint=False)
# thetas = np.linspace(0, 1, ntheta, endpoint=False)
# s = SurfaceRZFourier.from_wout(os.path.join(this_dir, "wout_loizu_qfm.nc"), quadpoints_phi=phis, quadpoints_theta=thetas)
# s.change_resolution(mpol, ntor)

s.extend_via_normal(extend_distance)

bs.set_points(s.gamma().reshape((-1, 3)))
Bbs = bs.B().reshape((nphi, ntheta, 3))
BdotN_surf = np.sum(Bbs * s.unitnormal(), axis=2) / np.linalg.norm(Bbs, axis=2)
pointData = {"B.n/B": BdotN_surf[:, :, None]}
s.to_vtk("initial_qfm_surface", extra_data=pointData)
# exit()

for i in range(1,n_iterations+1):
    # s = SurfaceRZFourier(dofs=s.dofs, mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
    # s.change_resolution(mpol, ntor)
    # bs.set_points(s.gamma().reshape((-1, 3)))
    
    qfm = QfmResidual(s, bs)
    qfm.J()
    
    vol = Volume(s)
    # vol_target = vol.J()
    qfm_surface = QfmSurface(bs, s, vol, vol_target)
    
    res = qfm_surface.minimize_qfm_penalty_constraints_LBFGS(tol=tol, maxiter=maxiter, constraint_weight=constraint_weight)
    print(f"#{i}: LBFGS ||vol constraint||={0.5*(s.volume()-vol_target)**2:.8e}, ||residual||={np.linalg.norm(qfm.J()):.8e}")
    
    # ar = Area(s)
    # # area_target = ar.J()
    # qfm_surface = QfmSurface(bs, s, ar, area_target)
    
    # res = qfm_surface.minimize_qfm_penalty_constraints_LBFGS(tol=tol, maxiter=maxiter, constraint_weight=constraint_weight)
    # print(f"#{i}: LBFGS ||area constraint||={0.5*(ar.J()-area_target)**2:.8e}, ||residual||={np.linalg.norm(qfm.J()):.8e}")    
    
    tol=tol/1.5
    maxiter+=5
if n_iterations>0 and do_SLSQP:
    res = qfm_surface.minimize_qfm_exact_constraints_SLSQP(tol=tol, maxiter=maxiter)
    print(f"#{i}: SLSQP ||vol constraint||={0.5*(s.volume()-vol_target)**2:.8e}, ||residual||={np.linalg.norm(qfm.J()):.8e}")

s.plot()

bs.set_points(s.gamma().reshape((-1, 3)))
Bbs = bs.B().reshape((nphi, ntheta, 3))
BdotN_surf = np.sum(Bbs * s.unitnormal(), axis=2) / np.linalg.norm(Bbs, axis=2)
pointData = {"B.n/B": BdotN_surf[:, :, None]}
s.to_vtk("final_qfm_surface", extra_data=pointData)

nphi_big   = np.max((256, nphi * 2 * s.nfp + 1))
ntheta_big = np.max((64, ntheta + 1))
surf_big = SurfaceRZFourier(dofs=s.dofs, nfp=s.nfp, mpol=s.mpol, ntor=s.ntor, quadpoints_phi=np.linspace(0, 1, nphi_big), quadpoints_theta=np.linspace(0, 1, ntheta_big), stellsym=s.stellsym)
bs.set_points(surf_big.gamma().reshape((-1, 3)))
Bbs = bs.B().reshape((nphi_big, ntheta_big, 3))
BdotN_surf = np.sum(Bbs * surf_big.unitnormal(), axis=2) / np.linalg.norm(Bbs, axis=2)
pointData = {"B.n/B": BdotN_surf[:, :, None]}
surf_big.to_vtk("final_qfm_surface_big", extra_data=pointData)

equil = Vmec(os.path.join(this_dir, "input.simple_nfp3"), verbose=False)
equil.indata.mpol = s.mpol+2
equil.indata.ntor = s.ntor+1
equil.indata.ns_array[:4]    = [    5,    16,    35,    51]
equil.indata.niter_array[:4] = [  200,   300,   400,  5000]
equil.indata.ftol_array[:4]  = [1e-10, 1e-11, 1e-11, 1e-14]
equil.indata.nfp = s.nfp
equil.indata.lasym = not s.stellsym
equil.boundary = qfm_surface.surface if n_iterations>0 else s
equil.write_input("input.loizu_qfm")