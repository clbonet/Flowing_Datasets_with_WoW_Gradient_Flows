# MMD with Riesz Kernel (MMDSW in the paper)
python main_dataset_distillation.py --init_true --method "mmdsw" --tgt_dataset "NormalizedMNIST" --n_epochs_wgd 20000 --ntry 3 --neval 5 --n_data_by_class 1
python main_dataset_distillation.py --init_true --method "mmdsw" --tgt_dataset "NormalizedMNIST" --n_epochs_wgd 20000 --ntry 3 --neval 5 --n_data_by_class 10
python main_dataset_distillation.py --init_true --method "mmdsw" --tgt_dataset "NormalizedMNIST" --n_epochs_wgd 20000 --ntry 3 --neval 5 --n_data_by_class 50

python main_dataset_distillation.py --init_true --method "mmdsw_aug" --tgt_dataset "NormalizedMNIST" --n_epochs_wgd 20000 --ntry 3 --neval 5 --n_data_by_class 1
python main_dataset_distillation.py --init_true --method "mmdsw_aug" --tgt_dataset "NormalizedMNIST" --n_epochs_wgd 20000 --ntry 3 --neval 5 --n_data_by_class 10
python main_dataset_distillation.py --init_true --method "mmdsw_aug" --tgt_dataset "NormalizedMNIST" --n_epochs_wgd 20000 --ntry 3 --neval 5 --n_data_by_class 50

python main_dataset_distillation.py --init_true --method "mmdsw_emb" --tgt_dataset "NormalizedMNIST" --n_epochs_wgd 20000 --ntry 3 --neval 5 --n_data_by_class 1
python main_dataset_distillation.py --init_true --method "mmdsw_emb" --tgt_dataset "NormalizedMNIST" --n_epochs_wgd 20000 --ntry 3 --neval 5 --n_data_by_class 10
python main_dataset_distillation.py --init_true --method "mmdsw_emb" --tgt_dataset "NormalizedMNIST" --n_epochs_wgd 20000 --ntry 3 --neval 5 --n_data_by_class 50

python main_dataset_distillation.py --init_true --method "mmdsw_aug_emb" --tgt_dataset "NormalizedMNIST" --n_epochs_wgd 20000 --ntry 3 --neval 5 --n_data_by_class 1
python main_dataset_distillation.py --init_true --method "mmdsw_aug_emb" --tgt_dataset "NormalizedMNIST" --n_epochs_wgd 20000 --ntry 3 --neval 5 --n_data_by_class 10
python main_dataset_distillation.py --init_true --method "mmdsw_aug_emb" --tgt_dataset "NormalizedMNIST" --n_epochs_wgd 20000 --ntry 3 --neval 5 --n_data_by_class 50



# DM
python main_dataset_distillation.py --init_true --method "dm" --tgt_dataset "NormalizedMNIST" --n_epochs_wgd 20000 --ntry 3 --neval 5 --n_data_by_class 1
python main_dataset_distillation.py --init_true --method "dm" --tgt_dataset "NormalizedMNIST" --n_epochs_wgd 20000 --ntry 3 --neval 5 --n_data_by_class 10
python main_dataset_distillation.py --init_true --method "dm" --tgt_dataset "NormalizedMNIST" --n_epochs_wgd 20000 --ntry 3 --neval 5 --n_data_by_class 50

python main_dataset_distillation.py --init_true --method "dm_aug" --tgt_dataset "NormalizedMNIST" --n_epochs_wgd 20000 --ntry 3 --neval 5 --n_data_by_class 1
python main_dataset_distillation.py --init_true --method "dm_aug" --tgt_dataset "NormalizedMNIST" --n_epochs_wgd 20000 --ntry 3 --neval 5 --n_data_by_class 10
python main_dataset_distillation.py --init_true --method "dm_aug" --tgt_dataset "NormalizedMNIST" --n_epochs_wgd 20000 --ntry 3 --neval 5 --n_data_by_class 50

python main_dataset_distillation.py --init_true --method "dm_emb" --tgt_dataset "NormalizedMNIST" --n_epochs_wgd 20000 --ntry 3 --neval 5 --n_data_by_class 1
python main_dataset_distillation.py --init_true --method "dm_emb" --tgt_dataset "NormalizedMNIST" --n_epochs_wgd 20000 --ntry 3 --neval 5 --n_data_by_class 10
python main_dataset_distillation.py --init_true --method "dm_emb" --tgt_dataset "NormalizedMNIST" --n_epochs_wgd 20000 --ntry 3 --neval 5 --n_data_by_class 50

python main_dataset_distillation.py --init_true --method "dm_aug_emb" --tgt_dataset "NormalizedMNIST" --n_epochs_wgd 20000 --ntry 3 --neval 5 --n_data_by_class 1
python main_dataset_distillation.py --init_true --method "dm_aug_emb" --tgt_dataset "NormalizedMNIST" --n_epochs_wgd 20000 --ntry 3 --neval 5 --n_data_by_class 10
python main_dataset_distillation.py --init_true --method "dm_aug_emb" --tgt_dataset "NormalizedMNIST" --n_epochs_wgd 20000 --ntry 3 --neval 5 --n_data_by_class 50


