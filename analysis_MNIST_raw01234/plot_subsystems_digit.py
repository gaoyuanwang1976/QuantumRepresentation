import qiskit.quantum_info as qi
import sys
import numpy as np
from qiskit.visualization.state_visualization import _bloch_multivector_data,plot_bloch_vector
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.feature_selection import mutual_info_regression
import matplotlib

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(6, 4))

beta=[0,0.5,1,1.5,2,2.5,3]

beta_plot=beta
ini=0.2

indices=np.loadtxt("indicies.dat")
digit_content_tmp=np.loadtxt("dim64_01234_label.dat")
digit_content=[]
for index in indices:
    digit_content.append(digit_content_tmp[int(index)])
digit_content=np.array(digit_content)

plt.subplots_adjust(top = 0.99, bottom=0.08, hspace=0.25, wspace=0.2)

cmap = matplotlib.colors.ListedColormap(["red","blue","yellow","green","grey"])
for b_index,b in enumerate([0,0.1]):
    qubit0=np.loadtxt(f"raw01234_latent_subs/ini{ini}_beta{b}_latent_qubit0.dat",dtype=np.complex128)
    qubit1=np.loadtxt(f"raw01234_latent_subs/ini{ini}_beta{b}_latent_qubit1.dat",dtype=np.complex128)
    dim=2

    num_data=int(len(qubit0))
    bloch_coordinates_final0=np.zeros((3,num_data))-100
    bloch_coordinates_final1=np.zeros((3,num_data))-100

    for index_x,x in enumerate(qubit0):
        dm=qi.DensityMatrix(np.reshape(x,(dim,dim)))
        assert(dm.is_valid())
        bloch_current=_bloch_multivector_data(dm)
        bloch_coordinates_final0[0][index_x]=bloch_current[0][0]
        bloch_coordinates_final0[1][index_x]=bloch_current[0][1]
        bloch_coordinates_final0[2][index_x]=bloch_current[0][2]

    for index_x,x in enumerate(qubit1):
        dm=qi.DensityMatrix(np.reshape(x,(dim,dim)))
        assert(dm.is_valid())
        bloch_current=_bloch_multivector_data(dm)
        bloch_coordinates_final1[0][index_x]=bloch_current[0][0]
        bloch_coordinates_final1[1][index_x]=bloch_current[0][1]
        bloch_coordinates_final1[2][index_x]=bloch_current[0][2]
    
    im=axs[b_index,0].scatter(bloch_coordinates_final0[0], bloch_coordinates_final0[2],c=digit_content,cmap=cmap,marker='o',alpha=0.5,label=f"digit",s=3)
    im=axs[b_index,1].scatter(bloch_coordinates_final1[0], bloch_coordinates_final1[2],c=digit_content,cmap=cmap,marker='o',alpha=0.5,label=f"digit",s=3)
    axs[b_index,0].tick_params(axis='y', labelsize=8)
    axs[b_index,1].tick_params(axis='y', labelsize=8)
    axs[b_index,0].tick_params(axis='x', labelsize=8)
    axs[b_index,1].tick_params(axis='x', labelsize=8)

cbar = fig.colorbar(im, ax=axs, orientation='vertical', fraction=0.02, pad=0.04)

cbar.ax.tick_params(size=0)
cbar.set_ticks([])

plt.show()