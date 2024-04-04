#!/usr/bin/env python3
import os
import sys
import time
from pathlib import Path
from simsopt.mhd import Vmec
from neat.fields import Simple
from neat.tracing import ChargedParticleEnsemble, ParticleEnsembleOrbit_Simple

def main(file, OUT_DIR=".", tfinal=1e-2, nparticles=3500):
    start_time = time.time()
    v = Vmec(file, verbose=False)
    g_particle = ChargedParticleEnsemble(r_initial=0.25)
    g_field = Simple(wout_filename=file, B_scale=5.7/v.wout.volavgB, Aminor_scale=1.7/v.wout.Aminor_p)
    g_orbits = ParticleEnsembleOrbit_Simple(g_particle,g_field,tfinal=tfinal,nparticles=nparticles,nsamples=2000,notrace_passing=0)
    total_time = time.time() - start_time
    try: os.remove('healaxis.dat')
    except: pass
    try: os.remove('start.dat')
    except: pass
    try: os.remove('fort.6601')
    except: pass
    print(f"  Running with {g_orbits.nparticles} particles took {total_time}s")
    print(f"  Loss fraction = {100*g_orbits.loss_fraction_array[-1]}% for a time of {tfinal}s")
    original_dir = os.getcwd()
    os.chdir(OUT_DIR)
    g_orbits.plot_loss_fraction(show=False, save=True)
    os.chdir(original_dir)

if __name__ == "__main__":
    # Create results folders if not present
    try:
        Path(sys.argv[2]).mkdir(parents=True, exist_ok=True)
        figures_results_path = str(Path(sys.argv[2]).resolve())
        main(sys.argv[1], sys.argv[2])
    except:
        main(sys.argv[1])