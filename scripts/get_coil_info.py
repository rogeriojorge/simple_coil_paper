#!/usr/bin/env python3
import sys
import numpy as np
from pathlib import Path
from simsopt import load
from simsopt.geo import (CurveLength, CurveCurveDistance, MeanSquaredCurvature, SurfaceRZFourier,
                         LpCurveCurvature, ArclengthVariation, CurveSurfaceDistance)
from simsopt.field import Current, Coil

def main(file, vmec_input=None):
    try:
        bs = load(file)
        coils = bs.coils
    except:
        try:
            [surfaces, base_curve, coils] = load(file)
        except:
            try:
                bs = load(file).Bfields[0]
                coils = bs.coils
            except:
                base_curves = load(file)
                base_currents = [Current(1) * 1e5]*len(base_curves)#, Current(1) * 1e5, Current(1) * 1e5, Current(1) * 1e5, Current(1) * 1e5, Current(1) * 1e5]
                coils = [Coil(curv, curr) for (curv, curr) in zip(base_curves, base_currents)]
                
    curves = [coils[i]._curve for i in range(len(coils))]
    currents = [coils[i].current.get_value() for i in range(len(coils))]
    # Jf = SquaredFlux(surf, bs, definition="local")
    Jls = [CurveLength(c) for c in curves]
    Jccdist = CurveCurveDistance(curves, 0, num_basecurves=len(curves))
    Jcs = [LpCurveCurvature(c, 2, 0) for i, c in enumerate(curves)]
    Jmscs = [MeanSquaredCurvature(c) for c in curves]
    Jals = [ArclengthVariation(c) for c in curves]

    try: outstr = f"C-C-Sep={Jccdist.shortest_distance():.2f}"
    except: outstr = ""
    cl_string = ", ".join([f"{j.J():.1f}" for j in Jls])
    kap_string = ", ".join(f"{np.max(c.kappa()):.1f}" for c in curves)
    msc_string = ", ".join(f"{j.J():.1f}" for j in Jmscs)
    outstr += f" lengths=sum([{cl_string}])={sum(j.J() for j in Jls):.1f}, curv=[{kap_string}],msc=[{msc_string}]"
    if vmec_input is not None:
        s = SurfaceRZFourier.from_vmec_input(vmec_input, range="full torus", nphi=64, ntheta=64)
        Jcsdist = CurveSurfaceDistance(curves, s, 0)
        try:    outstr += f", C-S-Sep={Jcsdist.shortest_distance():.2f}"
        except Exception as e: print(e);outstr += ""
    print(outstr)
    # print(f"Curve dofs={dir(curves[0])}")
    print(f"dofs local_full_dof_names = {curves[0].local_full_dof_names}")
    print(f"len(dofs) = {len(curves[0].x)}")
    print(f"currents = {currents}")

if __name__ == "__main__":
    if len(sys.argv) == 2:
        main(sys.argv[1])
    elif len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python get_coil_info.py [coil json file] [vmec_input]")