#!/bin/bash

# Xp with MMD + Riesz SW kernel
python main_transfer_learning.py --n_epochs_wgd 5000 --src_dataset MNIST --tgt_dataset FMNIST --k_shot 1 --ntry 3 --n_epochs_nn 50
python main_transfer_learning.py --n_epochs_wgd 5000 --src_dataset MNIST --tgt_dataset FMNIST --k_shot 5 --ntry 3 --n_epochs_nn 50 
python main_transfer_learning.py --n_epochs_wgd 5000 --src_dataset MNIST --tgt_dataset FMNIST --k_shot 10 --ntry 3 --n_epochs_nn 50
python main_transfer_learning.py --n_epochs_wgd 20000 --src_dataset MNIST --tgt_dataset FMNIST --k_shot 100 --ntry 3 --n_epochs_nn 50

python main_transfer_learning.py --n_epochs_wgd 5000 --src_dataset MNIST --tgt_dataset KMNIST --k_shot 1 --ntry 3 --n_epochs_nn 50
python main_transfer_learning.py --n_epochs_wgd 5000 --src_dataset MNIST --tgt_dataset KMNIST --k_shot 5 --ntry 3 --n_epochs_nn 50
python main_transfer_learning.py --n_epochs_wgd 5000 --src_dataset MNIST --tgt_dataset KMNIST --k_shot 10 --ntry 3 --n_epochs_nn 50
python main_transfer_learning.py --n_epochs_wgd 20000 --src_dataset MNIST --tgt_dataset KMNIST --k_shot 100 --ntry 3 --n_epochs_nn 50

python main_transfer_learning.py --n_epochs_wgd 5000 --src_dataset MNIST --tgt_dataset USPS --k_shot 1 --ntry 3 --n_epochs_nn 50
python main_transfer_learning.py --n_epochs_wgd 5000 --src_dataset MNIST --tgt_dataset USPS --k_shot 5 --ntry 3 --n_epochs_nn 50
python main_transfer_learning.py --n_epochs_wgd 5000 --src_dataset MNIST --tgt_dataset USPS --k_shot 10 --ntry 3 --n_epochs_nn 50
python main_transfer_learning.py --n_epochs_wgd 20000 --src_dataset MNIST --tgt_dataset USPS --k_shot 100 --ntry 3 --n_epochs_nn 50

# Xp with MMD + product of kernel
python main_transfer_learning.py --n_epochs_wgd 20000 --src_dataset MNIST --tgt_dataset FMNIST --k_shot 1 --ntry 3 --n_epochs_nn 50 --lr 10 --method "mmd_product"
python main_transfer_learning.py --n_epochs_wgd 20000 --src_dataset MNIST --tgt_dataset FMNIST --k_shot 5 --ntry 3 --n_epochs_nn 50 --lr 10 --method "mmd_product"
python main_transfer_learning.py --n_epochs_wgd 20000 --src_dataset MNIST --tgt_dataset FMNIST --k_shot 10 --ntry 3 --n_epochs_nn 50 --lr 10 --method "mmd_product"
python main_transfer_learning.py --n_epochs_wgd 20000 --src_dataset MNIST --tgt_dataset FMNIST --k_shot 100 --ntry 3 --n_epochs_nn 50 --lr 10 --method "mmd_product"

python main_transfer_learning.py --n_epochs_wgd 20000 --src_dataset MNIST --tgt_dataset KMNIST --k_shot 1 --ntry 3 --n_epochs_nn 50 --lr 10 --method "mmd_product"
python main_transfer_learning.py --n_epochs_wgd 20000 --src_dataset MNIST --tgt_dataset KMNIST --k_shot 5 --ntry 3 --n_epochs_nn 50 --lr 10  --method "mmd_product"
python main_transfer_learning.py --n_epochs_wgd 20000 --src_dataset MNIST --tgt_dataset KMNIST --k_shot 10 --ntry 3 --n_epochs_nn 50 --lr 10 --method "mmd_product"
python main_transfer_learning.py --n_epochs_wgd 20000 --src_dataset MNIST --tgt_dataset KMNIST --k_shot 100 --ntry 3 --n_epochs_nn 50 --lr 10 --method "mmd_product"

python main_transfer_learning.py --n_epochs_wgd 20000 --src_dataset MNIST --tgt_dataset USPS --k_shot 1 --ntry 3 --n_epochs_nn 50 --lr 10 --method "mmd_product"
python main_transfer_learning.py --n_epochs_wgd 20000 --src_dataset MNIST --tgt_dataset USPS --k_shot 5 --ntry 3 --n_epochs_nn 50 --lr 10 --method "mmd_product"
python main_transfer_learning.py --n_epochs_wgd 20000 --src_dataset MNIST --tgt_dataset USPS --k_shot 10 --ntry 3 --n_epochs_nn 50 --lr 10 --method "mmd_product"
python main_transfer_learning.py --n_epochs_wgd 20000 --src_dataset MNIST --tgt_dataset USPS --k_shot 100 --ntry 3 --n_epochs_nn 50 --lr 10 --method "mmd_product"

# Xp with OTDD
python main_transfer_learning.py --n_epochs_wgd 5000 --src_dataset MNIST --tgt_dataset FMNIST --k_shot 1 --ntry 3 --n_epochs_nn 50 --method "otdd"
python main_transfer_learning.py --n_epochs_wgd 5000 --src_dataset MNIST --tgt_dataset FMNIST --k_shot 5 --ntry 3 --n_epochs_nn 50 --method "otdd"
python main_transfer_learning.py --n_epochs_wgd 5000 --src_dataset MNIST --tgt_dataset FMNIST --k_shot 10 --ntry 3 --n_epochs_nn 50 --method "otdd"

python main_transfer_learning.py --n_epochs_wgd 5000 --src_dataset MNIST --tgt_dataset KMNIST --k_shot 1 --ntry 3 --n_epochs_nn 50 --method "otdd"
python main_transfer_learning.py --n_epochs_wgd 5000 --src_dataset MNIST --tgt_dataset KMNIST --k_shot 5 --ntry 3 --n_epochs_nn 50 --method "otdd"
python main_transfer_learning.py --n_epochs_wgd 5000 --src_dataset MNIST --tgt_dataset KMNIST --k_shot 10 --ntry 3 --n_epochs_nn 50 --method "otdd"
python main_transfer_learning.py --n_epochs_wgd 5000 --src_dataset MNIST --tgt_dataset KMNIST --k_shot 100 --ntry 3 --n_epochs_nn 50 --method "otdd"

python main_transfer_learning.py --n_epochs_wgd 5000 --src_dataset MNIST --tgt_dataset USPS --k_shot 1 --ntry 3 --n_epochs_nn 50 --method "otdd"
python main_transfer_learning.py --n_epochs_wgd 5000 --src_dataset MNIST --tgt_dataset USPS --k_shot 5 --ntry 3 --n_epochs_nn 50 --method "otdd"
python main_transfer_learning.py --n_epochs_wgd 5000 --src_dataset MNIST --tgt_dataset USPS --k_shot 10 --ntry 3 --n_epochs_nn 50 --method "otdd"


