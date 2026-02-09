#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 12:00:15 2021.

@author: spiros
"""
import os
os.environ["KERAS_BACKEND"] = "torch"

import sys
import copy
import torch
import keras
import pickle
import pathlib

from opt import make_masks
from opt import custom_train_loop_torch
from opt import get_model
from opt import get_data
from opt import get_model_name

device = torch.device(f"cuda:{int(sys.argv[1])}" if torch.cuda.is_available() else "cpu")

# Run parameters
save = True
seq_flag = False if int(sys.argv[2]) == 0 else True
noise = True
early_stop = False if int(sys.argv[3]) == 0 else True
trial = int(sys.argv[4])
keras.utils.set_random_seed(trial)

rfs_type = 'somatic'
model_type = int(sys.argv[5])
sparse = False
input_sample = None
if model_type == 0:
    # dendritic ANN (dANN) with random connections
    conventional = False
    rfs = False
elif model_type == 1:
    # dendritic ANN (dANN) with global RFs
    conventional = False
    rfs = True
elif model_type == 2:
    # dendritic ANN (dANN) with local RFs
    conventional = False
    rfs = True
    rfs_type = 'dendritic'
elif model_type == 3:
    # vanilla ANN (vANN) all-to-all connections (no RFs)
    conventional = True
    rfs = False
elif model_type == 4:
    # vanilla ANN (vANN) with random connections
    conventional = True
    rfs = False
    sparse = True
elif model_type == 5:
    # vanilla ANN (vANN) with global RFs
    conventional = True
    rfs = True
elif model_type == 6:
    # vanilla ANN (vANN) with local RFs
    conventional = True
    rfs = True
    rfs_type = 'dendritic'
elif model_type == 7:
    # sparse ANN (sANN)
    conventional = False
    rfs = False
    sparse = True
elif model_type == 8:
    # sparse ANN (sANN) with global RFs
    conventional = False
    rfs = True
    sparse = True
elif model_type == 9:
    # sparse ANN (sANN) with local RFs
    conventional = False
    rfs = True
    sparse = True
    rfs_type = 'dendritic'
elif model_type == 10:
    # dendritic ANN (dANN) with all-to-all inputs
    conventional = False
    rfs = True
    sparse = False
    input_sample = 'all_to_all'
elif model_type == 11:
    # sparse ANN (sANN) with all-to-all inputs
    conventional = False
    rfs = True
    sparse = True
    input_sample = 'all_to_all'

sigma = float(sys.argv[6])
# Get the data
datatype = sys.argv[7]
batch_size = 128
data, labels, img_height, img_width, channels = get_data(
    validation_split=0.1,
    dtype=datatype,
    normalize=True,
    add_noise=noise,
    sigma=sigma,
    sequential=seq_flag,
    batch_size=batch_size,
    seed=trial,
)

# Extract the data in train, validation, and test sets
x_train, x_val, x_test = data['train'], data['val'], data['test']
y_train, y_val, y_test = labels['train'], labels['val'], labels['test']

# Model architectures
num_dends, num_soma = int(sys.argv[8]), int(sys.argv[9])
num_classes = len(set(y_train))
num_layers = int(sys.argv[10])
dends = num_layers*[num_dends]
soma = num_layers*[num_soma]
synapses = int(sys.argv[11])

# Build the masks
Masks = make_masks(
    dends,
    soma,
    synapses,
    num_layers,
    img_width,
    img_height,
    num_classes,
    channels,
    conventional=conventional,
    rfs=rfs,
    rfs_type=rfs_type,
    rfs_mode='random',
    seed=trial,
)

# Get the model
input_shape = (img_width * img_height * channels, )

fname_model = get_model_name(
    conventional,
    rfs,
    sparse,
    rfs_type,
    input_sample
)

drop_flag = False if int(sys.argv[12]) == 0 else True
rate_of_drop = float(sys.argv[13])
# Set the foldername extension
if seq_flag:
    file_tag = "_sequential"
else:
    file_tag = ""

# Change the model name if dropout
if drop_flag:
    fname_model += f"_dropout_{rate_of_drop}"

# Get the model
model = get_model(
    input_shape,
    num_layers,
    dends,
    soma,
    num_classes,
    fname_model=fname_model,
    dropout=drop_flag,
    rate=rate_of_drop,
)

# Attach the model to device
model.to(device)

# Apply the masks to initial weights
PARAMS = model.get_weights()
PARAMSmod = [PARAMS[i]*Masks[i] for i in range(len(PARAMS))]

# Set the initial weights by zeroing out not connected nodes.
model.set_weights(PARAMSmod)
model_untrained = copy.deepcopy(model)

# Instantiate the optimizer and the loss function
lr = float(sys.argv[14])
optimizer = keras.optimizers.Adam(learning_rate=lr)
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=False)

if lr != 0.001:
    file_tag += f"_lr_{lr}"

# Hyperparameters
if datatype == 'mnist':
    num_epochs = 15 if not seq_flag else 30
elif datatype == 'fmnist':
    num_epochs = 25 if not seq_flag else 50
elif datatype == 'kmnist':
    num_epochs = 25 if not seq_flag else 50
elif datatype == 'emnist':
    num_epochs = 50 if not seq_flag else 100
elif datatype == 'cifar10':
    num_epochs = 50 if not seq_flag else 100

if early_stop:
    num_epochs = 100

print(f"\nModel: {fname_model}, trial: {trial}, layers: {num_layers}, "
      f"noise: {sigma}, dataset: {datatype}, tag: {file_tag}\n")

Masks = [torch.tensor(m).to(device) for m in Masks]

# train and evaluate the model
model, out = custom_train_loop_torch(
    model,
    loss_fn,
    optimizer,
    Masks,
    batch_size,
    num_epochs,
    x_train, y_train,
    x_val, y_val,
    x_test, y_test,
    shuffle=False if seq_flag else True,
    early_stop=early_stop,
    patience=10,
)

# Store masks in the output dictionary
out['Masks'] = Masks

if save:
    # the local directory to save the data
    path_to_dir_local = sys.argv[15]
    if not os.path.exists(path_to_dir_local):
        os.mkdir(path_to_dir_local)

    # subdirectory with name of model, num of layers and other tags added
    sub_tag = f"results_{datatype}_{num_layers}_layer{file_tag}/"
    dirname = pathlib.Path(f'{path_to_dir_local}/{sub_tag}')
    if not os.path.exists(dirname):
        os.mkdir(dirname)

    # Create the output directory
    outdir_name = pathlib.Path(f"{dirname}/{fname_model}")
    if not os.path.exists(outdir_name):
        os.mkdir(outdir_name)

    # Save the model
    postfix = f"sigma_{sigma}_trial_{trial}_dends_{num_dends}_soma_{num_soma}"
    # Save the untrained and trained model
    # model_untrained.save(pathlib.Path(f"{outdir_name}/untrained_model_{postfix}.h5"))
    model.save(pathlib.Path(f"{outdir_name}/model_{postfix}.keras"))

    # Save the results
    fname_res = pathlib.Path(f"{outdir_name}/results_{postfix}.pkl")
    with open(fname_res, 'wb') as handle:
        pickle.dump(out, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\nResults have been saved in: {dirname}/{fname_model}")
