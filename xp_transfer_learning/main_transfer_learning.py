import jax
import argparse
import sys

import numpy as np

from utils_eval import train_nn

sys.path.append("../")
from lib.datasets import get_dataset
from lib.gd_images import wasserstein_gradient_descent
from lib.mmd import target_value_and_grad_riesz
from lib.baseline_mmd_product_space import mmd_product_flow
from lib.classif_nn import eval_nn
from lib.utils_labels import get_labels
from lib.baseline_otdd_jax import otdd_flow


parser = argparse.ArgumentParser()
parser.add_argument("--method", type=str, default="hierarchical_mmd", help="Method to use")
parser.add_argument("--ntry", type=int, default=3, help="Number of try")
parser.add_argument("--neval", type=int, default=5, help="Number of evaluations of NNs")
parser.add_argument("--path_data", type=str, default="~/torch_datasets", help="Directory torch data")
parser.add_argument("--path_results", type=str, default="./results", help="Directory to save the results")
parser.add_argument("--src_dataset", type=str, default="MNIST", help="Target dataset")
parser.add_argument("--tgt_dataset", type=str, default="FMNIST", help="Target dataset")
parser.add_argument("--n_data_by_class", type=int, default=200, help="Number of data by class for synthetic dataset")
parser.add_argument("--k_shot", type=int, default=5, help="Number of data by class for target dataset")
parser.add_argument("--batch_size", type=int, default=256, help="Batch size training NN")
parser.add_argument("--n_epochs_wgd", type=int, default=1000, help="Number of epochs for WGD")
parser.add_argument("--n_epochs_nn", type=int, default=50, help="Number of epochs to train NN")
parser.add_argument("--lr_wgd", type=float, default=1, help="Learning rate for WGD")
parser.add_argument("--m_wgd", type=float, default=0.9, help="Momentum for WGD")
args = parser.parse_args()

if __name__ == "__main__":
    # if args.method == "otdd":
    #     from lib.baseline_otdd_pytorch import otdd_flow

    # Load dataset
    rng = jax.random.PRNGKey(0)
    master_key, key_src, key_tgt = jax.random.split(rng, num=3)

    path_data = args.path_data
    src_dataset = args.src_dataset
    tgt_dataset = args.tgt_dataset

    X_data_src, y_src, _, _ = get_dataset(key_src, src_dataset, args.n_data_by_class, path_data)
    X_data_tgt, y_tgt, X_test, y_test = get_dataset(key_tgt, tgt_dataset, args.k_shot, path_data)

    L_acc = np.zeros((args.ntry, args.neval))
    L_acc_baseline = np.zeros((args.ntry, args.neval))

    for i in range(args.ntry):
        master_key, key_wgd = jax.random.split(master_key)

        lr = args.lr_wgd
        m = args.m_wgd

        # Get synthetic images
        if args.method == "hierarchical_mmd":
            @jax.jit
            def target_value_and_grad(x, y, rng):
                return target_value_and_grad_riesz(
                    x, y, rng, n_sample_batch=args.k_shot
                )

            L_loss, xk = wasserstein_gradient_descent(
                X_data_src, X_data_tgt, target_value_and_grad, key_wgd,
                n_epochs=args.n_epochs_wgd, lr=lr, m=m
                )
            xk = np.clip(xk, 0, 1)

            yk = get_labels(X_data_tgt, y_tgt, xk).reshape(xk.shape[0], -1)

        elif args.method == "otdd":
            xk, yk, L_loss = otdd_flow(key_wgd, X_data_src, y_src, X_data_tgt, y_tgt,
                                       n_epochs=args.n_epochs_wgd)
            # xk, yk, L_loss = otdd_flow(X_data_src, y_src, X_data_tgt, y_tgt,
            #                            n_epochs_wgd=args.n_epochs_wgd)

        elif args.method == "mmd_product":
            xk, yk, L_loss = mmd_product_flow(key_wgd, X_data_src, y_src,
                                              X_data_tgt, y_tgt,
                                              lr=lr, m=m,
                                              n_epochs=args.n_epochs_wgd)

        # Evaluate synthetic images by training neural networks on it
        for j in range(args.neval):
            master_key, key_train_nn = jax.random.split(master_key)

            # Train on reduced dataset
            model_baseline = train_nn(key_train_nn, X_data_tgt, y_tgt,
                                      n_epochs=args.n_epochs_nn)
            acc_baseline = eval_nn(model_baseline, X_test, y_test)

            L_acc_baseline[i, j] = acc_baseline

            # Train on augmented dataset
            X_augmented = np.concatenate([X_data_tgt, xk], axis=1)
            y_augmented = np.concatenate([y_tgt, yk], axis=1)

            model = train_nn(key_train_nn, X_augmented, y_augmented,
                             n_epochs=args.n_epochs_nn)
            acc_augmented = eval_nn(model, X_test, y_test)
            L_acc[i, j] = acc_augmented

    path_acc = "/accuracy_" + src_dataset + "_to_" + tgt_dataset +\
               "_" + args.method + "_k" + str(args.k_shot) +\
               "_nepochsnn" + str(args.n_epochs_nn)

    np.savetxt(args.path_results + path_acc, L_acc, delimiter=",")

    path_imgs = "/synth_images_" + src_dataset + "_to_" + tgt_dataset +\
                "_" + args.method + "_k" + str(args.k_shot) +\
                "_nepochsnn" + str(args.n_epochs_nn)

    np.save(args.path_results + path_imgs, xk)

    path_loss = "/loss_" + src_dataset + "_to_" + tgt_dataset +\
                "_" + args.method + "_k" + str(args.k_shot) +\
                "_nepochsnn" + str(args.n_epochs_nn)

    np.savetxt(args.path_results + path_loss, L_loss, delimiter=",")

    path_acc_baseline = "/accuracy_baseline_" + src_dataset + "_to_" +\
                        tgt_dataset + "_k" + str(args.k_shot) +\
                        "_nepochsnn" + str(args.n_epochs_nn)

    np.savetxt(args.path_results + path_acc_baseline, L_acc_baseline,
               delimiter=",")
