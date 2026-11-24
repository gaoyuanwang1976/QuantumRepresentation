try:
    from qiskit.extensions import UnitaryGate
except ModuleNotFoundError:
    from qiskit.circuit.library import UnitaryGate
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.circuit.library import RawFeatureVector
from qiskit import QuantumCircuit,QuantumRegister,ClassicalRegister #, Aer, execute
import numpy as np
# import core

import preprocessing

def rx_kernel(qc,x_params,n_external_inputs,n_extra_qubits):
    for i in range(n_external_inputs):
        qc.rx(x_params[i], i)
    for j in range(n_extra_qubits):
        qc.h(n_external_inputs+j)


def rx_circuit(x_params,n_inputs,n_extra_qubits,n_output):
    qc=QuantumCircuit(n_inputs,n_output)
    state_vector=Statevector(qc)
    qc.initialize(state_vector,list(range(0,n_inputs)))
    qc.barrier()
    rx_kernel(qc,x_params,n_inputs,n_extra_qubits)
    qc.barrier()
    qc.measure(list(range(n_output)),list(range(n_output)))
    return qc

def ising_rxx_rxz_interaction(qc,x_params,theta,n_inputs,n_layers,n_external_inputs,n_extra_qubits):
    typ=isinstance(theta,list)
    param_index=0
    for layer in range(n_layers):
        for i in range(n_external_inputs):
            qc.rx(x_params[i], i)
        for j in range(n_extra_qubits):
            qc.h(n_external_inputs+j)

        for q2 in range(1,n_inputs):
            qc.rzz(theta[param_index],0,q2)
            param_index+=1

        for q2 in range(1,n_inputs):
            qc.rzx(theta[param_index],0,q2)
            param_index+=1

        for q_tmp in range(n_inputs):
            qc.ry(theta[param_index],q_tmp)
            param_index+=1
 
    for i in range(n_external_inputs):
        qc.rx(x_params[i], i)
    for j in range(n_extra_qubits):
        qc.h(n_external_inputs+j)

    return qc



def ising_interaction(qc,x_params,theta,n_layers,n_qubit):
    param_index=0
    for layer in range(n_layers):
        #for i in range(n_feature):
        #    qc.rx(x_params[i], i)

        for q1 in range(n_qubit):
            for q2 in range(q1,n_qubit):
                if q1!=q2:
                    qc.rzz(theta[param_index],q1,q2)
                    param_index+=1

        for q_tmp in range(n_qubit):
            qc.ry(theta[param_index],q_tmp)
            param_index+=1
 
    for i in range(n_qubit):
        qc.rx(x_params[i], i)
    for i in range(n_qubit):
        qc.rx(x_params[i+2], i)

    return qc

def ising_interaction_noInput(qc,theta,n_layers,n_qubit,num_auxiliary):
    typ=isinstance(theta,list)
    param_index=0
    num_qubit_total=n_qubit+num_auxiliary
    for layer in range(n_layers):

        for q1 in range(num_qubit_total):
            for q2 in range(q1,num_qubit_total):
                if q1!=q2:
                    qc.rzz(theta[param_index],q1,q2)
                    param_index+=1

        for q_tmp in range(num_qubit_total):
            qc.ry(theta[param_index],q_tmp)
            param_index+=1

    return qc

def mnist_circuit(circuit,parameters,n_layers,n_qubit):
    '''
    Encoder from tech blog 'Quantum Autoencoder with MNIST classification' by Fernanda et al.
    '''
    num_per_layer=10
    for layer in range(n_layers):
        #Applying a ry gate in each qubit
        for i in range(n_qubit):
            #the rotation of the ry gate is defined in the parameters list
            #based on the layer
            circuit.ry(parameters[(layer*num_per_layer)+i], i)
        circuit.barrier() #Create a barrier

        circuit.cx(2,0) #Apply a CNOT gate between the qubit 2 and 0
        circuit.cx(3,1) #Apply a CNOT gate between the qubit 3 and 1
        circuit.cx(5,4) #Apply a CNOT gate between the qubit 5 and 4
        circuit.barrier() #Create a barrier
        
        #Apply a RY gate in the qubit 0 with the rotation specified in the parameter list
        circuit.ry(parameters[6+(layer*num_per_layer)],0)
        #Apply a RY gate in the qubit 1 with the rotation specified in the parameter list
        circuit.ry(parameters[7+(layer*num_per_layer)],1)
        #Apply a RY gate in the qubit 4 with the rotation specified in the parameter list
        circuit.ry(parameters[8+(layer*num_per_layer)],4)
        circuit.barrier() #Create a barrier
        
        circuit.cx(4,1) #Apply a CNOT gate between the qubit 4 and 1
        circuit.barrier() #Create a barrier
        
        #Apply a RY gate in the qubit 1 with the rotation specified in the parameter list
        circuit.ry(parameters[9+(layer*num_per_layer)], 1)
        circuit.barrier() #Create a barrier
    return circuit

def mnist_cz_circuit(circuit,parameters,n_layers,n_qubit,n_trash):
    '''
    Used for encoder type 'ef_qae'
    from paper 'Quantum autoencoders with enhanced data encoding' by Carlos Bravo-Prieto
    '''
    # assuming 2 trash_qubits for now
    num_per_layer = n_trash * n_qubit
    if (n_trash == 1):
        for layer in range(n_layers):
            #Applying a ry gate in each qubit
            for i in range(n_qubit):
                #the rotation of the ry gate is defined in the parameters list
                #based on the layer
                circuit.ry(parameters[(layer*num_per_layer)+i], i)
            circuit.barrier() #Create a barrier

            for t in range(1, n_qubit):
                circuit.cz(0, t)
            circuit.barrier() #Create a barrier

    if (n_trash == 2):
        for layer in range(n_layers):
            #Applying a ry gate in each qubit
            for i in range(n_qubit):
                #the rotation of the ry gate is defined in the parameters list
                #based on the layer
                circuit.ry(parameters[(layer*num_per_layer)+i], i)
            circuit.barrier() #Create a barrier

            circuit.cz(0, 1) #Apply CZ between trash qubits
            
            for t in range(2, n_qubit, 2):
                circuit.cz(0, t)
                if t+1 < n_qubit:
                    circuit.cz(1, t+1)
            circuit.barrier() #Create a barrier

            # second cascade
            for i in range(n_qubit):
                #the rotation of the ry gate is defined in the parameters list
                #based on the layer
                circuit.ry(parameters[(layer*num_per_layer)+n_qubit+i], i)
            circuit.barrier() #Create a barrier

            circuit.cz(1, 0) #Apply CZ between trash qubits
            
            for t in range(2, n_qubit, 2):
                circuit.cz(1, t)
                if t+1 < n_qubit:
                    circuit.cz(0, t+1)
            circuit.barrier() #Create a barrier
    #final RY on trash qubits
    for t_i in range(n_trash):
        circuit.ry(parameters[n_layers * num_per_layer + t_i], t_i)
    return circuit

def ansatz(num_qubits, reps):
    return RealAmplitudes(num_qubits, reps=reps)

def auto_encoder_circuit(num_latent, num_trash):
    qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
    cr = ClassicalRegister(1, "c")
    circuit = QuantumCircuit(qr, cr)
    circuit.compose(ansatz(num_latent + num_trash, 5), range(0, num_latent + num_trash), inplace=True)
    circuit.barrier()
    auxiliary_qubit = num_latent + 2 * num_trash
    # swap test
    circuit.h(auxiliary_qubit)
    for i in range(num_trash):
        circuit.cswap(auxiliary_qubit, num_latent + i, num_latent + num_trash + i)

    circuit.h(auxiliary_qubit)
    circuit.measure(auxiliary_qubit, cr[0])
    return circuit