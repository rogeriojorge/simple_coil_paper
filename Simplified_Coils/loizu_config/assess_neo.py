#!/usr/bin/env python3
import os
import numpy as np
import shutil
import subprocess
from simsopt.mhd import Vmec
import matplotlib.pyplot as plt

if not os.path.isfile("boozmn_loizu_qfm.nc"):
    booz_input_file = f'in_booz.final'
    boozxform_executable = '/Users/rogeriojorge/bin/xbooz_xform'
    bashCommand = f'{boozxform_executable} {booz_input_file}'
    run_booz = subprocess.Popen(bashCommand.split())
    run_booz.wait()

if not os.path.isfile("neo_out.loizu_qfm"):
    neo_input_file = f'neo_in.loizu_qfm'
    neo_executable = '/Users/rogeriojorge/bin/xneo'
    bashCommand = f'{neo_executable} loizu_qfm'
    run_neo = subprocess.Popen(bashCommand.split())
    run_neo.wait()

token = open('neo_out.loizu_qfm','r')
linestoken=token.readlines()
eps_eff=[]
s_radial=[]
for x in linestoken:
    s_radial.append(float(x.split()[0])/50)
    eps_eff.append(float(x.split()[1])**(2/3))
token.close()
s_radial = np.array(s_radial)
eps_eff = np.array(eps_eff)
s_radial = s_radial[np.argwhere(~np.isnan(eps_eff))[:,0]]
eps_eff = eps_eff[np.argwhere(~np.isnan(eps_eff))[:,0]]
fig = plt.figure(figsize=(7, 3), dpi=200)
ax = fig.add_subplot(111)
plt.plot(s_radial,eps_eff, '*-', label=f'eps eff')
ax.set_yscale('log')
plt.xlabel(r'$s=\psi/\psi_b$', fontsize=12)
plt.ylabel(r'$\epsilon_{eff}$', fontsize=14)
plt.xlim([0,1])

plt.tight_layout()
fig.savefig(f'neo_out.pdf', dpi=fig.dpi)#, bbox_inches = 'tight', pad_inches = 0)