import qiskit.quantum_info as qi
import sys
import numpy as np
from qiskit.visualization.state_visualization import _bloch_multivector_data,plot_bloch_vector
import matplotlib.pyplot as plt
import math
from sklearn.feature_selection import mutual_info_regression
from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit import ParameterVector


num_aux=1
dim=2
n_layers=3
n_qubit=int(1+num_aux)

entanglement_choice='linear'
n_qubit_measure=1
observables = SparsePauliOp.from_sparse_list(
                [("Z" * n_qubit_measure, range(n_qubit_measure), 1)],
                num_qubits=n_qubit_measure,
            )

projection=observables[0]


def sigmoid(x):
  return ((1 / (1 + math.exp(-x*10)))-0.5)*2

def objective(params,dataset,target_set):
    total_loss=0
    index=0

    ansatz = QuantumCircuit(n_qubit)
    tmp_n_params=math.comb(n_qubit,2)     #number of gates for ising_interaction (zz) embedding, this number may change for another embedding
    n_params = (n_qubit+tmp_n_params)*n_layers
    theta = ParameterVector("theta_par", n_params)

    param_index=0

    for layer in range(n_layers):
        for q1 in range(n_qubit):
            for q2 in range(q1,n_qubit):
                if q1!=q2:
                    ansatz.rzz(theta[param_index],q1,q2)
                    param_index+=1

        for q_tmp in range(n_qubit):
            ansatz.ry(theta[param_index],q_tmp)
            param_index+=1

    real_ansatz = RealAmplitudes(num_qubits=n_qubit,reps=n_layers,insert_barriers=True,entanglement=entanglement_choice,flatten=True)
    bound_circuit = real_ansatz.assign_parameters(dict(zip(real_ansatz.parameters, params)))
    
    qc = QuantumCircuit(n_qubit,1)
    qc.compose(bound_circuit, inplace=True)


    for data,target in zip(dataset,target_set):
        
        evolved_state=data.evolve(qc)

        reduced_state=qi.partial_trace(evolved_state,list(range(n_qubit-1)))
        predict_raw=reduced_state.expectation_value(projection)
        predict=sigmoid(predict_raw)
        
        total_loss+=abs(predict-target)*abs(predict-target)*1./len(dataset)
        index+=1
    return total_loss  

beta=[0,0.1,0.2,0.3,0.4]
initial=[-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4]
beta_plot=beta

indices=np.loadtxt("indicies.dat")
digit_content_tmp=np.loadtxt("dim64_01234_label.dat")
digit_content=[]
for index in indices:
    digit_content.append(digit_content_tmp[int(index)])

digit_content=(np.array(digit_content)*1.-2)/2

random_dataset=[]
for _ in range(len(digit_content)):
    random_qubit=qi.random_density_matrix(dims=dim).data.flatten()
    random_dataset.append(random_qubit)



num_epochs=10
test_ansatz = RealAmplitudes(num_qubits=n_qubit,reps=n_layers,insert_barriers=True,entanglement=entanglement_choice,flatten=True)

num_params=len(test_ansatz.parameters)

for b_index,b in enumerate(beta):


    for ini_index,ini in enumerate(initial):
        qubit0=np.loadtxt(f"raw01234_latent_subs/ini{ini}_beta{b}_latent_qubit0.dat",dtype=np.complex128)
        qubit1=np.loadtxt(f"raw01234_latent_subs/ini{ini}_beta{b}_latent_qubit1.dat",dtype=np.complex128)



        for qubit_index,qubit in enumerate([qubit0,qubit1]):

            S_list = []
            for index_x,x in enumerate(qubit):
                dm=qi.DensityMatrix(np.reshape(x,(dim,dim)))
                dim_aux=2**num_aux
                state = [0]*dim_aux
                state[0]=1
                zero_state=qi.DensityMatrix(state)
                expand_state=zero_state.expand(dm)
                S_list.append(expand_state)

            initial_weight=[0]*num_params
            best_loss = np.inf
            best_weights = initial_weight
            patience=0
            for epoch in range(num_epochs):
                res = minimize(objective, best_weights, args=(S_list,digit_content),method='COBYLA', options={'maxiter':100, 'disp':False,'rhobeg':1})

                current_loss = res.fun

                # Save best weights and loss
                if current_loss < best_loss:
                    best_loss = current_loss
                    best_weights = res.x

                if epoch>0 and abs(prev_loss - current_loss) < 1e-6:
                    patience+=1
                if patience>5:
                    print(f"qubit index {qubit_index}, {b},{ini},Epoch {epoch+1}/{num_epochs} | Loss: {current_loss:.6f} | Best loss: {best_loss:.6f}")
                    print("Early stopping")
                    break
                prev_loss=current_loss
                if epoch==9:
                    print(f"qubit index {qubit_index}, {b},{ini},Epoch {epoch+1}/{num_epochs} | Loss: {current_loss:.6f} | Best loss: {best_loss:.6f}")

        print('\n')
    print('end of this beta\n')