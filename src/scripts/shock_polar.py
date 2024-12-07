"""
The first figure
"""

import numpy as np
import matplotlib.pyplot as plt

import paths
import helpers

OUTFILE = paths.figures / "shock_polar.pdf"
M1 = [1.2, 1.5, 2, 3,5, 10,1e3]
NAMES = ['1.2','1.5','2','3','5','10','10^3']
GAMMA = 5/3

plt.style.use('bmh')

if __name__ == "__main__":
    fig = plt.figure(figsize=(7,4))
    ax = fig.add_subplot(1,1,1)
    ax.set_aspect('equal')
    # x = np.linspace(0, 1, 1000)
    colors = plt.cm.viridis(np.linspace(0, 1, len(M1)))
    for m1,c,n in zip(M1, colors,NAMES):
        xmin = helpers.alpha_sq(m1**2, GAMMA)/m1**2
        x = np.linspace(xmin, 1, 1000)
        yp, ym = helpers.y(x, m1**2, GAMMA)
        p_valid = ~np.isnan(yp)
        n_valid = ~np.isnan(ym)
        _x = np.concatenate([x[p_valid][::-1], x[n_valid]])
        _y = np.concatenate([yp[p_valid][::-1], ym[n_valid]])
        ax.plot(_x, _y,label=f'$M_1 = {n}$', color=c)
        # ax.plot(x, ym, color=c)
    ylims = ax.get_ylim()
    phi_max = np.arcsin(1/GAMMA)
    for k in [1,-1]:
        x = np.linspace(0, 1, 100)
        ax.plot(x, k*np.tan(phi_max)*x, '--', color='k')
    ax.text(0.2, 0.2*np.tan(phi_max)+0.05, '$\\psi_{{\\rm max},M_1\\rightarrow \\infty}$', fontsize=14,rotation=phi_max*180/np.pi,va='center',ha='center')
    ax.text(0.62, -0.15, '$\\psi_{{\\rm max}}$', fontsize=14,rotation=phi_max*180/np.pi,va='center',ha='center')

    xmin = helpers.alpha_sq(M1[-1]**2, GAMMA)/M1[-1]**2
    eta = np.linspace(xmin, 1, 1000)
    
    m1 = 1/np.linspace(1e-5,1-1e-5,1000)
    _dat = np.array([helpers.get_psi_max(_m1**2, GAMMA) for _m1 in m1])
    psi_max = _dat[:,0]
    eta_max = _dat[:,1]
    ax.plot(eta_max*np.cos(psi_max), eta_max*np.sin(psi_max), 'k:')
    ax.plot(eta_max*np.cos(psi_max), -eta_max*np.sin(psi_max), 'k:')
    
    
    ax.set_ylim(ylims)
    ax.set_xlabel('$u_2/u_1 \\cos \\psi$',fontsize=14)
    ax.set_ylabel('$u_2/u_1 \\sin \\psi$',fontsize=14)
    _=ax.legend(prop={'size': 14},loc=(.15,1.2),ncol=2)
    ax.text(0.05,-0.35,'$\\gamma = 5/3$', fontsize=14,rotation=0,va='center',ha='center')
    fig.savefig(OUTFILE, bbox_inches='tight')