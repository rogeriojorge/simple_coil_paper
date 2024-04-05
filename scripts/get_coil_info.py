#!/usr/bin/env python3
import sys
import numpy as np
from pathlib import Path
from simsopt import load
from simsopt.geo import (CurveLength, CurveCurveDistance, MeanSquaredCurvature, LpCurveCurvature, ArclengthVariation)

def main(file, OUT_DIR="."):
    try:
        bs = load(file)
        coils = bs.coils
    except:
        try:
            [surfaces, base_curve, coils] = load(file)
        except:
            bs = load(file).Bfields[0]
            coils = bs.coils
    curves = [coils[i]._curve for i in range(len(coils))]
    currents = [coils[i].current.get_value() for i in range(len(coils))]
    # Jf = SquaredFlux(surf, bs, definition="local")
    Jls = [CurveLength(c) for c in curves]
    Jccdist = CurveCurveDistance(curves, 0, num_basecurves=len(curves))
    Jcs = [LpCurveCurvature(c, 2, 0) for i, c in enumerate(curves)]
    Jmscs = [MeanSquaredCurvature(c) for c in curves]
    Jals = [ArclengthVariation(c) for c in curves]

    outstr = f"C-C-Sep={Jccdist.shortest_distance():.2f}"
    cl_string = ", ".join([f"{j.J():.1f}" for j in Jls])
    kap_string = ", ".join(f"{np.max(c.kappa()):.1f}" for c in curves)
    msc_string = ", ".join(f"{j.J():.1f}" for j in Jmscs)
    outstr += f" lengths=sum([{cl_string}])={sum(j.J() for j in Jls):.1f}, curv=[{kap_string}],msc=[{msc_string}]"
    print(outstr)
    # print(f"Curve dofs={dir(curves[0])}")
    print(f"dofs local_full_dof_names = {curves[0].local_full_dof_names}")
    print(f"len(dofs) = {len(curves[0].x)}")
    print(f"currents = {currents}")

if __name__ == "__main__":
    # Create results folders if not present
    try:
        Path(sys.argv[2]).mkdir(parents=True, exist_ok=True)
        figures_results_path = str(Path(sys.argv[2]).resolve())
        main(sys.argv[1], sys.argv[2])
    except:
        main(sys.argv[1])