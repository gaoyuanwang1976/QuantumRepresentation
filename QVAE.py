from pyexpat.errors import XML_ERROR_RESERVED_PREFIX_XML
from re import A
import numpy as np
from qiskit.circuit import ParameterVector
from qiskit.algorithms.optimizers import COBYLA, SPSA, ADAM
import qiskit.quantum_info as qi
from qiskit.utils import algorithm_globals
# algorithm_globals.random_seed = 0
import math
import core
import preprocessing
from math import comb
import time as time
import copy
import matplotlib.pyplot as plt
from pathlib import Path

import os
abspath = os.path.abspath('__file__')
dname = os.path.dirname(abspath)
os.chdir(dname)
import sys
import argparse
import ast
import pickle

#for performance metrics
from sklearn.metrics import confusion_matrix, roc_curve, auc

# %% parsing
class ModelState:
    def __init__(self, fit_result, best_fit_result, patience, epochs_left):
        self.fit_result = fit_result
        self.best_fit_result = best_fit_result
        self.patience_counter = patience
        self.epochs_left = epochs_left

def create_train_test_set(args):
    # get useful args
    import_name = args.import_data
    shuffle=args.shuffle
    shuffleseed = args.shuffleseed
    alternate = True
    partition_size=args.partition_size
    if partition_size != 'max':
        parition_size = int(partition_size)
    ratio = args.partition_ratio.split(":")
    ratio = [float(entry) for entry in ratio]

    dataset = preprocessing.import_dataset(import_name,'classical',shuffle, shuffleseed)
    maxData = preprocessing.get_max_data(import_name)
    minData = preprocessing.get_min_data(import_name)

    if alternate:
        dataset = preprocessing.alternate_g(dataset)
        print("LENGTH: ", len(dataset))
        dataset = preprocessing.get_uq_g(dataset)
    print(f"using dataset of length {len(dataset)}")
    if partition_size != 'max':
        partition_split = int(partition_size)
    else:
        partition_split=len(dataset)
    print(f'using partition size of {partition_split}')
    train_set, test_set = preprocessing.train_test(dataset, partition_split, ratio)
    train_len = len(train_set)
    print("for training:")
    preprocessing.get_info_g(train_set, True)
    print("for testing:")
    preprocessing.get_info_g(test_set, True)
    test_len = len(test_set)

    Xtrain, ytrain = preprocessing.convert_for_qiskit_classical(train_set)
    Xtest, ytest = preprocessing.convert_for_qiskit_classical(test_set)

    Xtrain=preprocessing.normalize_amplitude(Xtrain)
    Xtest=preprocessing.normalize_amplitude(Xtest)

    ### convert input into density matrices ###
    Xtrain_dm=[]
    Xtest_dm=[]

    if args.global_state == True:
        combined_state_tr=np.zeros((len(Xtrain[0]),len(Xtrain[0])))
        combined_state_te=np.zeros((len(Xtest[0]),len(Xtest[0])))

    valid = True
    for i,x in enumerate(Xtrain):
        dm = qi.DensityMatrix(x)
        Xtrain_dm.append(qi.DensityMatrix(x).data)
        if args.global_state == True:
            combined_state_tr=combined_state_tr+Xtrain_dm[-1]*1./len(Xtrain)

    for x in Xtest:
        Xtest_dm.append(qi.DensityMatrix(x).data)
        if args.global_state == True:
            combined_state_te=combined_state_te+Xtest_dm[-1]*1./len(Xtest)

    if args.global_state == True:
        Xtrain_origin=copy.deepcopy(Xtrain_dm)
        Xtest_origin=copy.deepcopy(Xtest_dm)
        if reconstruction_loss=='wasserstein':
            Xtrain_pool=[]
            Xtest_pool=[]
            tr_eigenvalues, tr_eigenvectors = np.linalg.eig(combined_state_tr)
            for tr_i,tr_eigenvalue in enumerate(tr_eigenvalues):
                count=int(round(tr_eigenvalue.real,2)*100)
                for i in range(count):
                    Xtrain_pool.append(qi.DensityMatrix(tr_eigenvectors[tr_i]).data)

            te_eigenvalues, te_eigenvectors = np.linalg.eig(combined_state_te)
            for te_i,te_eigenvalue in enumerate(te_eigenvalues):
                count=int(round(te_eigenvalue.real,2)*100)
                for i in range(count):
                    Xtest_pool.append(qi.DensityMatrix(te_eigenvectors[te_i]).data)
            Xtrain=np.array(Xtrain_pool)
            Xtest=np.array(Xtest_pool)
        else:
            Xtrain=np.array([combined_state_tr])
            Xtest=np.array([combined_state_te])

    else:
        Xtrain=np.array(Xtrain_dm)
        Xtest=np.array(Xtest_dm)

    return Xtrain, ytrain, Xtest, ytest

def create_model(args, only_model=False, only_qnn=False):
    '''
    Create model from arguments. It's possible to start from saved model
    return model, best_model, best_train_score, patience_counter, num_epoch
    UPDATES args.reconstruction_loss if invalid
    '''
    import_path = args.import_path
    num_epoch=args.epochs
    parsed_shots=args.shots
    n_layers = args.num_layers
    decoder_inv = args.decoder_inv
    cost_type = args.cost_type
    opt_iter = args.opt_iter

    resume_dir = args.resume_dir
    
    ######### QVAE specific parameters
    trash_qubits=list(range(args.num_trash_qubits))

    if cost_type not in ['latent','trash']:
        cost_type='latent'
        print('cost type choice not recognized, using latent')

    reconstruction_loss=args.reconstruction_loss
    if reconstruction_loss not in ['cross_entropy','fidelity','wasserstein','JSD']:
        reconstruction_loss='fidelity'
        args.reconstruction_loss = 'fidelity'
        print('reconstruction loss choice not recognized, using fidelity loss')

    beta_weight=args.beta_weight

    regularizer_type=args.regularizer_type
    if regularizer_type not in ['KLD','JSD','fidelity']:
        print('divergence type not recognized, use JSD instead')
        regularizer_type='JSD'

    num_auxiliary_encoder=args.num_auxiliary_encoder
    num_auxiliary_decoder=args.num_auxiliary_decoder

    ##########

    if args.optimizer.lower() == 'cobyla':
        optimizer = COBYLA
        rhobeg = args.rhobeg
    elif args.optimizer.lower() == 'spsa':
        optimizer = SPSA
    elif args.optimizer.lower() == 'adam':
        optimizer = ADAM
    else:
        print("problem with parsing optimizer, defaulting to COBYLA")
        optimizer = COBYLA
        rhobeg = args.rhobeg
    

#######################
##### circuit def #####
#######################

    n_features = args.input_dim
    n_qubit = math.log2(n_features)
    assert(int(n_qubit)==n_qubit)
    n_qubit=int(n_qubit)
    x_params = ParameterVector('x',n_features)
    input_task=args.input_task
    circuit_type=args.circuit_type
    if input_task=='gene':
        # for encoder
        n_qubit_e=n_qubit+num_auxiliary_encoder

        tmp_gates_e=comb(n_qubit_e,2)     #number of gates for ising_interaction (zz) embedding, this number may change for another embedding
        n_gates_e = (n_qubit_e+tmp_gates_e)*n_layers         

        # for decoder
        n_qubit_d=n_qubit+num_auxiliary_decoder
        tmp_gates_d=comb(n_qubit_d,2)
        n_gates_d = (n_qubit_d+tmp_gates_d)*n_layers

        theta_params = ParameterVector('theta', n_gates_e+n_gates_d)

        num_encoder_params=n_gates_e
    elif input_task=='mnist':
        assert(n_qubit==6 or n_qubit==4)
        output_shape = (int(math.sqrt(n_features)), int(math.sqrt(n_features)))
        if circuit_type == 'ef_qae':
            n_gates=2 * n_qubit * n_layers + 2
        elif circuit_type == 'real_amp':
            n_gates=(n_layers+1)*n_qubit
        if not decoder_inv:
            theta_params = ParameterVector('theta', n_gates+n_gates)
        else:
            theta_params = ParameterVector('theta', n_gates)
        num_encoder_params=n_gates

    qc_e=core.encoder(n_layers,n_qubit,theta_params[:num_encoder_params],num_auxiliary_encoder,input_task,len(trash_qubits),circuit_type)
    # print circuit
    # qc_e.decompose().draw(output='mpl', filename='encoder_circuit.png')
    if not decoder_inv:
        qc_d=core.decoder(n_layers,n_qubit,theta_params[num_encoder_params:],num_auxiliary_decoder,input_task,len(trash_qubits),circuit_type)
    else:
        qc_d=qc_e.inverse()

    qnn = core.QVAE_NN(circuit=qc_e, encoder=qc_e,decoder=qc_d,input_params=x_params, weight_params=theta_params,num_encoder_params=num_encoder_params,decoder_inv=decoder_inv,trash_qubits=trash_qubits,num_auxiliary_encoder=num_auxiliary_encoder,num_auxiliary_decoder=num_auxiliary_decoder)
    if only_qnn:
        return qnn

    if args.initial_point!=None:
        initial_point=[float(args.initial_point)]*qnn.num_weights
    else:
        initial_point=args.initial_point
    if optimizer == COBYLA:
        print(f"Using COBYLA with rhobeg={rhobeg}")
        opt_alg = optimizer(maxiter=opt_iter, rhobeg=rhobeg)
    else:
        opt_alg = optimizer(maxiter=opt_iter)
    
    model=core.QVAE_trainer(neural_network=qnn,optimizer=opt_alg,initial_point=initial_point,loss= 'squared_error',warm_start=True,reconstruction_loss=reconstruction_loss,beta=beta_weight,regularizer_type=regularizer_type,cost_type=cost_type)


    best_train_score=-sys.maxsize
    if resume_dir is not None:
        print("Using saved weights")
        with open(os.path.join(resume_dir,'model_state.pkl'), "rb") as f:
            model_state = pickle.load(f)
            model._fit_result = model_state.fit_result
            
            best_model = copy.deepcopy(model) 
            best_model._fit_result = model_state.best_fit_result
            if only_model:
                return model, best_model, None, None, None

            print(f"From loaded state, train score {model.score(Xtrain, Xtrain, reconstruction_loss)}")
            best_train_score = best_model.score(Xtrain, Xtrain,reconstruction_loss)
            print(f"From loaded state, best train score {best_train_score}")
            
            patience_counter = model_state.patience_counter
            num_epoch = model_state.epochs_left
            print(f"From loaded state, patience_counter={patience_counter}, epochs left = {num_epoch}")
        return model, best_model, best_train_score, patience_counter, num_epoch
    else:
        return model, None, best_train_score, 0, num_epoch

def reconstruct_args(file_path):
    found = False
    with open(file_path, "r") as f:
        for line in f:
            if line.startswith("hyper parameters:"):
                args_str = line.split("hyper parameters:  Namespace", 1)[1].strip().strip("()")
                args_str = args_str.split(", ")
                found = True
                break
        if not found:
            raise ValueError("No hyper parameters found")

    kv_pairs = [p.split("=") for p in args_str]
    literal_dict = "{"
    for i, kv in enumerate(kv_pairs):
        literal_dict += f"'{kv[0]}': {kv[1]}"
        if i < len(kv_pairs)-1: 
            literal_dict += ", "
        else:
            literal_dict += "}"
    args_dict = ast.literal_eval(literal_dict)
    return argparse.Namespace(**args_dict)

def restore_model_from_dir(dir_path, try_resume=False):
    slurm_file = ""
    for f in os.listdir(dir_path):
        if f.startswith("slurm"):
            if slurm_file == "":
                slurm_file = f
            else:
                print(f"Warning: found multiple slurm output files, using {slurm_file}")
    if slurm_file == "":
        print(f"Error: no slurm output files found")
        exit(1)

    args = reconstruct_args(os.path.join(dir_path, slurm_file))

    # We effectively want to restore from best model
    args.resume_dir = dir_path if try_resume else None # TODO: not backwards compatible, older training runs don't have resume_dir, or dirs don't have model.pkl
    model, best_model, _, _, _ = create_model(args, True)

    return model, best_model

def restore_qnn_from_dir(dir_path, try_resume=False):
    slurm_file = ""
    for f in os.listdir(dir_path):
        if f.startswith("slurm"):
            if slurm_file == "":
                slurm_file = f
            else:
                print(f"Warning: found multiple slurm output files, using {slurm_file}")
    if slurm_file == "":
        print(f"Error: no slurm output files found")
        exit(1)

    args = reconstruct_args(os.path.join(dir_path, slurm_file))

    # We effectively want to restore from best model
    args.resume_dir = dir_path if try_resume else None # TODO: not backwards compatible, older training runs don't have resume_dir, or dirs don't have model.pkl
    qnn = create_model(args, only_qnn=True)

    return qnn 

def restore_weights_from_dir(dir_path):
    weights_file = os.path.join(dir_path, "best_weights.dat")
    if not os.path.exists(weights_file):
        print(f"no best_weights file found, trying weights.dat")
        weights_file = os.path.join(dir_path, "weights.dat")
        if not os.path.exists(weights_file):
            print(f"Error: no weights file found.")
            exit(1)

    weights = np.loadtxt(weights_file)

    return weights

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Simulate a QNN with the appropriate hyperparameters.")

    ## training parameters
    parser.add_argument('-e','--epochs', required=False, type=int, help='the desired number of epochs to run', default=10)
    parser.add_argument('-p','--patience', required=False, type=int, help='upper limit for the patience counter used in validation', default=5)
    parser.add_argument('-i', '--import_data', required=True, help='name of the input files')
    parser.add_argument('--partition_size', required=False, help='sets partition size for splitting data into train and test sets (scales the partition_ratio arg)', default='max')
    parser.add_argument('--partition_ratio', required=False, type=str, help="governs the ration of partition sizes in the training and test sets. a list of the form [train, test]", default="0.7:0.3")
    parser.add_argument('--shuffle', action='store_true', help='determines whether to shuffle data before alternating', default=False)
    parser.add_argument('--shuffleseed', required=False, type=int, help='a seed for use in shuffling the dataset, if left False and --shuffle=True, will be completely random', default=False)
    parser.add_argument('--global_state', action='store_true', help='whether a global quantum state is used')
    parser.add_argument('--initial_point', required=False, help='initial parameter of the neural network', default=None)
    parser.add_argument('--input_task',required=False, help='specify whether gene expression task or MNIST64',default='gene')

    ## model parameters
    parser.add_argument('--import_path', required=True, help='path to the input file')
    parser.add_argument('-x','--shots', required=False, type=int, help="the number of shots per circuit simulation", default=100)
    parser.add_argument('--num_layers', required = False, type=int, help='determines the number of alternating layers in the circuit', default=1)
    parser.add_argument('--circuit_type',required=False, help='circuit type for encoder,ef_qae or real_amp',default='real_amp')
    parser.add_argument('--decoder_inv',action='store_true', help='whether the decoder should be the inverse of the encoder')
    parser.add_argument('--cost_type',required=False, help='whether the cost should be calculated based on latent or trash fidelity',default="latent")
    parser.add_argument('--resume_dir',required=False,help="if resuming training, provide dir to pickle file for model and best model",default=None)
    parser.add_argument('--opt_iter',required=False,type=int,help='number of iterations per epoch for optimizer',default=100)
    parser.add_argument('-t','--num_trash_qubits', required=False, type=int, help='number of trash qubits, the first 0,...,N-1 qubits will be traced out in the latent space', default=1)
    parser.add_argument('--reconstruction_loss', required=False, type=str, help='define the loss used in the reconstruction term of the objective', default='fidelity')
    parser.add_argument('--beta_weight', required=False, type=float, help='the beta parameter that controlls the relative weight of the quantum entropy term in the objective', default=1.0)
    parser.add_argument('--regularizer_type', required=False, type=str, help='choose between KL-Divergence and JS-Divergence', default='JSD')
    parser.add_argument('--num_auxiliary_encoder', required=False, type=int, help='number of auxiliary qubits in the encoder', default=0)
    parser.add_argument('--num_auxiliary_decoder', required=False, type=int, help='number of auxiliary qubits in the decoder', default=0)
    parser.add_argument('-o','--optimizer', required=False, type=str, help='determines the Qiskit optimizer used in qnn', default='cobyla')
    parser.add_argument('--rhobeg', required=False, type=float, help="rhobeg param for cobyla optimizer", default=1.0)

    ## plotting/output parameters
    parser.add_argument('--output_dir',required=False, help='output directory for reconstructed state and latent state',default="./tmp/")
    parser.add_argument('--input_dim', required=False, type=int, help='customize the input data dimension, if zero the original input dimension is preserved', default=0)


    args = parser.parse_args()
    print("hyper parameters: ",args)
    output_dir = args.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    ## create train and test set
    Xtrain, ytrain, Xtest, ytest = create_train_test_set(args)

    if args.input_dim==0:
        args.input_dim=len(Xtrain[0])

    input_dim=args.input_dim
    ## Create model
    model, best_model, best_train_score, patience_counter, num_epoch = create_model(args) 
    if patience_counter >= args.patience:
        print("Patience already reached, exiting")
        exit()

    s_time = time.localtime()
    print("Started training at {}/{}/{}, {}:{}:{}".format(s_time[1], s_time[2], s_time[0], s_time[3], s_time[4], s_time[5]), flush=True)
    train_fids = []
    train_obj_vals = []
    for epoch in range(num_epoch):
        print('start fitting')
        model.fit(Xtrain, Xtrain)

        this_train_score=model.score(Xtrain, Xtrain,args.reconstruction_loss)

        this_train_fidelity=model.score(Xtrain, Xtrain,'fidelity')
        train_fids.append(this_train_fidelity)

        train_obj_vals.extend(model.obj_fun_vals)

        print(epoch,'train fidelity:',this_train_fidelity,'train score:',this_train_score, flush=True)

        if this_train_score > best_train_score: #validation wrapper
            best_train_epoch = epoch
            best_train_score=this_train_score
            best_model= copy.deepcopy(model)
            patience_counter = 0
            print(f"new best train score {best_train_score}")
        else:
            patience_counter+=1
        if patience_counter == args.patience:
            print("ran out of patience")
            break
        model_state = ModelState(model._fit_result, best_model._fit_result, patience_counter, num_epoch - epoch - 1)
        with open(os.path.join(output_dir,'model_state.pkl'), "wb") as f:
            pickle.dump(model_state, f)

    f_time = time.localtime()
    print("Finished training at {}/{}/{}, {}:{}:{}".format(f_time[1], f_time[2], f_time[0], f_time[3], f_time[4], f_time[5]), flush=True)
    trainscore = best_model.score(Xtrain, Xtrain,'fidelity')
    testscore = best_model.score(Xtest, Xtest,'fidelity')
    print(f'best model train score: {trainscore}')
    print(f'best model test score: {testscore}')

    filename = os.path.join(output_dir, f'train_fidelity_{args.input_task}{input_dim}.png')
    print(f'saving {filename}')
    plt.title("Train fidelity against iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Avg train fidelity")
    plt.plot(range(len(train_fids)), train_fids)
    plt.savefig(filename)
    plt.close()

    filename = os.path.join(output_dir, f'train_obj_fun_{args.input_task}{input_dim}.png')
    print(f'saving {filename}')
    plt.title("Train objective function against iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Avg train objective function")
    plt.plot(range(len(train_obj_vals)), train_obj_vals)
    plt.savefig(filename)

    if output_dir!=None:
        if args.global_state==True:
            output_test_tmp,latent_test_tmp,trash_test_tmp=best_model.predict(Xtest_origin)
            output_train_tmp,latent_train_tmp,trash_train_tmp=best_model.predict(Xtrain_origin)

        else:     
            output_test_tmp,latent_test_tmp,trash_test_tmp=best_model.predict(Xtest)
            output_train_tmp,latent_train_tmp,trash_train_tmp=best_model.predict(Xtrain)
        output_train=[]
        latent_train=[]
        output_test=[]
        latent_test=[]
        for train_o,train_l,y in zip(output_train_tmp,latent_train_tmp,ytrain):
            output_train.append(np.append(train_o.data.flatten(),y))
            latent_train.append(np.append(train_l.data.flatten(),y))

        for idx, (test_o,test_l,y) in enumerate(zip(output_test_tmp,latent_test_tmp,ytest)):
            output_test.append(np.append(test_o.data.flatten(),y))
            latent_test.append(np.append(test_l.data.flatten(),y))
 
        np.savetxt(os.path.join(output_dir,'weights.dat'),model.weights)
        np.savetxt(output_dir+'/best_weights.dat',best_model.weights)
        np.savetxt(output_dir+'/reconstruct_train.dat', output_train)
        np.savetxt(output_dir+'/latent_train.dat', latent_train)
        np.savetxt(output_dir+'/reconstruct_test.dat', output_test)
        np.savetxt(output_dir+'/latent_test.dat', latent_test)
        np.savetxt(output_dir+'/latent_all.dat', np.concatenate((latent_train, latent_test)))
        np.savetxt(output_dir+'/reconstruct_all.dat', np.concatenate((output_train, output_test)))


