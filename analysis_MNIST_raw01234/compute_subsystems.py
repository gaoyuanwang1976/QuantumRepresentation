import qiskit.quantum_info as qi
import sys
import numpy as np

dim=4
beta=[0.1,0.2,0.3,0.5,0.6,0.7,0.8,0.9,1.5,2]
initial=[-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4]
for b in beta:
    mi_list=[]
    for ini in initial:
        two_qubit_latent=np.loadtxt(f"raw01234_latent_states/ini{ini}_beta{b}_latent_all.dat",dtype=np.complex128).T[:-1].T
        X_DM=[]
        sub_0=[]
        sub_1=[]
        for data_i in two_qubit_latent:
            x_dm=np.reshape(data_i,newshape=(dim,dim))
            sub_0.append(qi.partial_trace(x_dm, [0]).data.flatten())     
            sub_1.append(qi.partial_trace(x_dm, [1]).data.flatten())

        
        np.savetxt(f"raw01234_latent_subs/ini{ini}_beta{b}_latent_qubit0.dat",sub_0)
        np.savetxt(f"raw01234_latent_subs/ini{ini}_beta{b}_latent_qubit1.dat",sub_1)