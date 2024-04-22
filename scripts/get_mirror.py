#!/usr/bin/env python3
import os
import sys
from pathlib import Path
from qi_functions import MirrorRatioPen
from simsopt.mhd import Vmec, QuasisymmetryRatioResidual

def main(file, OUT_DIR="."):
    vmec = Vmec(file, verbose=False)
    print(f"Mirror Delta={MirrorRatioPen(vmec,t=0)}")

    quasisymmetry_target_surfaces = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    absolute_path = os.path.abspath(file)
    if 'QH' in absolute_path: QA_or_QH = 'QH'
    else: QA_or_QH = 'QA'
    qs = QuasisymmetryRatioResidual(vmec, quasisymmetry_target_surfaces, helicity_m=1, helicity_n=-1 if QA_or_QH == 'QH' else 0)
    print(f"Quasisymmetry objective: {qs.total()}")
    print(f"Magnetic well : {vmec.vacuum_well()}")
    print(f"Aspect ratio: {vmec.aspect()}")

if __name__ == "__main__":
    # Create results folders if not present
    try:
        Path(sys.argv[2]).mkdir(parents=True, exist_ok=True)
        figures_results_path = str(Path(sys.argv[2]).resolve())
        main(sys.argv[1], sys.argv[2])
    except:
        main(sys.argv[1])