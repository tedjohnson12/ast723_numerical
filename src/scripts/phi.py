"""
The second figure
"""

import numpy as np
import matplotlib.pyplot as plt

import paths
import helpers

OUTFILE = paths.figures / "phi.pdf"
M1 = [1.2, 1.5, 2, 3,5, 10,1e3]
NAMES = ['1.2','1.5','2','3','5','10','10^3']
GAMMA = 5/3

plt.style.use('bmh')

if __name__ == "__main__":
    fig = plt.figure(figsize=(7,4))
    ax = fig.add_subplot(1,1,1)
    # ax.set_aspect('equal')
    colors = plt.cm.viridis(np.linspace(0, 1, len(M1)))
    for m1,c,n in zip(M1, colors,NAMES):
        psi_max, eta_max = helpers.get_psi_max(m1**2, GAMMA)
        psi = np.linspace(0, psi_max, 1000)[1:-1]
        eta_weak = np.array([helpers.eta_weak(_psi, psi_max, eta_max, m1**2, GAMMA) for _psi in psi])
        eta_strong = np.array([helpers.eta_strong(_psi,psi_max, eta_max, m1**2, GAMMA) for _psi in psi])
        res_weak = ~np.isnan(eta_weak)
        res_strong = ~np.isnan(eta_strong)
        eta = np.concatenate([eta_weak[res_weak], eta_strong[res_strong][::-1]])
        psi = np.concatenate([psi[res_weak], psi[res_strong][::-1]])
        tan_phi = (1-eta*np.cos(psi))/(eta*np.sin(psi))
        phi = np.arctan(tan_phi)
        ax.plot(psi, phi, label=f'$M_1 = {n}$', color=c)
    ax.set_xlabel('$\\psi$',fontsize=14)
    ax.set_ylabel('$\\phi$',fontsize=14)
    _=ax.legend(prop={'size': 14},loc=(.15,1.1),ncol=2)
    # ax.set_xlim(-0.05*np.pi, 1.05*np.pi/2)
    ax.text(0.55,0.15,'$\\gamma = 5/3$', fontsize=14,rotation=0,va='center',ha='center')
    
    m1 = 1/np.linspace(1e-5,1-1e-5,1000)
    _dat = np.array([helpers.get_psi_max(_m1**2, GAMMA) for _m1 in m1])
    psi_max = _dat[:,0]
    eta_max = _dat[:,1]
    tan_phi = (1-eta_max*np.cos(psi_max))/(eta_max*np.sin(psi_max))
    phi = np.arctan(tan_phi)
    ax.plot(psi_max, phi, 'k:')
    
    fig.savefig(OUTFILE, bbox_inches='tight')