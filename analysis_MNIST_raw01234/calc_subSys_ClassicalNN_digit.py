import qiskit.quantum_info as qi
import sys
import numpy as np
from qiskit.visualization.state_visualization import _bloch_multivector_data,plot_bloch_vector
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.feature_selection import mutual_info_regression

from qiskit import QuantumCircuit
from qiskit.circuit.library import EfficientSU2
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,accuracy_score,roc_auc_score
from sklearn.ensemble import RandomForestRegressor

final=[]
'''
class MultiClassNN(nn.Module):
    def __init__(self, input_dim=3, num_classes=5):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(64, num_classes)
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        f = self.feature(x)
        logits = self.classifier(f)
        return logits / self.temperature

'''
class MultiClassNN(nn.Module):
    def __init__(self, input_dim=3, num_classes=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.net(x)
#'''

beta=[0,0.1,0.2,0.3,0.4]
initial=[-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4]
beta_plot=beta

indices=np.loadtxt("indicies.dat")
digit_content_tmp=np.loadtxt("dim64_01234_label.dat")
digit_content=[]
for index in indices:
    digit_content.append(digit_content_tmp[int(index)])

digit_content=np.array(digit_content)
average_main=[]
average_other=[]
num_qubits=2
dim=2

random_dataset=[]
for _ in range(len(digit_content)):
    random_qubit=qi.random_density_matrix(dims=dim).data.flatten()
    random_dataset.append(random_qubit)


for b_index,b in enumerate(beta):
    current_main=[]
    current_less=[]
    for ini_index,ini in enumerate(initial):
        current_sub=[]
        qubit0=np.loadtxt(f"raw01234_latent_subs/ini{ini}_beta{b}_latent_qubit0.dat",dtype=np.complex128)
        qubit1=np.loadtxt(f"raw01234_latent_subs/ini{ini}_beta{b}_latent_qubit1.dat",dtype=np.complex128)

        num_data=len(qubit0)

        bloch_coordinates_final=np.zeros((num_qubits,3,num_data))-100
        for qubit_index,qubit in enumerate([qubit0,qubit1]):
            for index_x,x in enumerate(qubit):
                dm=qi.DensityMatrix(np.reshape(x,(dim,dim)))

                bloch_current=_bloch_multivector_data(dm)
                bloch_coordinates_final[qubit_index][0][index_x]=bloch_current[0][0]
                bloch_coordinates_final[qubit_index][1][index_x]=bloch_current[0][1]
                bloch_coordinates_final[qubit_index][2][index_x]=bloch_current[0][2]
            

            current_x=bloch_coordinates_final[qubit_index][0]
            current_y=bloch_coordinates_final[qubit_index][2]


            current_x = torch.tensor(current_x,dtype=torch.float32).unsqueeze(1)
            current_y = torch.tensor(current_y,dtype=torch.float32).unsqueeze(1)
            target_list=[digit_content]
            for target_index,target in enumerate(target_list):
                target = torch.tensor(target,dtype=torch.float32).unsqueeze(1)
                r = torch.sqrt(current_x**2 + current_y**2)
                X = torch.cat((current_x, current_y, r), dim=1)
                
                X = (X - X.mean(dim=0)) / (X.std(dim=0) + 1e-8)
                y = target
                y = y.squeeze().long()

                model = MultiClassNN()
                criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
                optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
                scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)

                epochs = 400
                for epoch in range(epochs):
                    optimizer.zero_grad()
                    y_pred = model(X)
                    loss = criterion(y_pred, y)
                    
                    loss.backward()
                    optimizer.step()

                with torch.no_grad():
                    logits = model(X)
                    preds = torch.argmax(logits, dim=1)

                
                accu = accuracy_score(
                    y.cpu().numpy(),
                    preds.cpu().numpy()
)
                
                current_sub.append(accu)
                
        current_main.append(min(current_sub))
        current_less.append(max(current_sub))

    average_main.append(np.mean(current_main))
    average_other.append(np.mean(current_less))

    final.append([b,np.mean(current_main),np.std(current_main),np.mean(current_less),np.std(current_less)])
final=np.array(final)
x= final.T[0]
plt.plot(x,final.T[1])
plt.plot(x,final.T[3])
plt.errorbar(x,final.T[1],final.T[2])
plt.errorbar(x,final.T[3],final.T[4])
plt.show()
            
        