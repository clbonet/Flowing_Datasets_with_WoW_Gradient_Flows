#!/bin/bash

python domain_adaptation_evolution.py --n_epochs_wgd 500000 --src_dataset FMNIST --ntry 3 --pretrained --save_every 5000
python domain_adaptation_evolution.py --n_epochs_wgd 500000 --src_dataset KMNIST --ntry 3 --pretrained --save_every 5000
python domain_adaptation_evolution.py --n_epochs_wgd 500000 --src_dataset USPS --ntry 3 --pretrained --save_every 5000

python domain_adaptation_evolution_cifar.py --n_epochs_wgd 500000 --src_dataset SVHN --ntry 3 --save_every 5000 --pretrained --n_data_by_class 100
