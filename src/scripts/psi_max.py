"""
An extra figure I like
"""

import numpy as np
import matplotlib.pyplot as plt

import paths
import helpers

OUTFILE = paths.figures / "psi_max.pdf"
M1 = [1.2, 1.5, 2, 3,5, 10,1e3]
NAMES = ['1.2','1.5','2','3','5','10','10^3']
GAMMA = 5/3

plt.style.use('bmh')

if __name__ == "__main__":
    fig = plt.figure(figsize=(7,4))
    ax = fig.add_subplot(1,1,1)

    ax.set_xlabel('$M_1$',fontsize=14)
    ax.set_ylabel('$\\psi_{\\rm max}$',fontsize=14)
    
    m1 = 1/np.linspace(1e-2,1-1e-5,1000)
    _dat = np.array([helpers.get_psi_max(_m1**2, GAMMA) for _m1 in m1])
    psi_max = _dat[:,0]
    
    ax.plot(m1, psi_max, 'k')
    ax.set_xscale('log')
    ax.set_xticks([1,10,100])
    ax.set_xticklabels([1,10,100])
    ax.axhline(np.arcsin(1/GAMMA), ls='--', color='k')
        
    fig.savefig(OUTFILE, bbox_inches='tight')