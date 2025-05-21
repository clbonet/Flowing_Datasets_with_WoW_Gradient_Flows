import jax
import argparse
import sys

import numpy as np

from utils_eval import train_nn
from distribution_matching import target_value_and_grad_dm_full_class_emb, \
    target_value_and_grad_dm_full_class_emb_mnist, \
    target_value_and_grad_dm_full_class_ambient, \
    target_value_and_grad_dm_full_class_only_emb, \
    target_value_and_grad_dm_full_class_only_aug_mnist, \
    target_value_and_grad_dm_full_class_only_aug
from distribution_matching_mmd_sw import target_value_and_grad_riesz_dataset_distillation, target_value_and_grad_riesz_dataset_distillation_aug_mnist, target_value_and_grad_riesz_dataset_distillation_aug_full
from distribution_matching_mmd_sw_no_emb import target_value_and_grad_riesz_dataset_distillation_no_emb_aug_mnist, target_value_and_grad_riesz_dataset_distillation_no_emb_aug_full

sys.path.append("../")
from lib.datasets import get_dataset
from lib.gd_images import wasserstein_gradient_descent
from lib.mmd import target_value_and_grad_riesz
from lib.classif_nn import eval_nn
from lib.utils_labels import get_labels


parser = argparse.ArgumentParser()
parser.add_argument("--method", type=str, default="hierarchical_mmd",
                    help="Method to use")
parser.add_argument("--ntry", type=int, default=5, help="Number of try")
parser.add_argument("--neval", type=int, default=1,
                    help="Number of evaluations of NNs")
parser.add_argument("--path_data", type=str, default="~/torch_datasets",
                    help="Directory torch data")
parser.add_argument("--path_results", type=str, default="./results",
                    help="Directory to save the results")
parser.add_argument("--tgt_dataset", type=str, default="MNIST",
                    help="Target dataset")
parser.add_argument("--n_data_by_class", type=int, default=5,
                    help="Number of data by class for synthetic dataset")
parser.add_argument("--batch_size_wgd", type=int, default=256,
                    help="Batch size WGD")
parser.add_argument("--n_epochs_wgd", type=int, default=1000,
                    help="Number of epochs for WGD")
parser.add_argument("--lr_wgd", type=float, default=1,
                    help="Learning rate for WGD")
parser.add_argument("--m_wgd", type=float, default=0.9,
                    help="Momentum for WGD")
parser.add_argument("--init_true", help="If True, initialize from real data",
                    action="store_true")
args = parser.parse_args()

if __name__ == "__main__":
    # Load dataset
    rng = jax.random.PRNGKey(0)
    master_key, key_tgt = jax.random.split(rng)

    path_data = args.path_data
    tgt_dataset = args.tgt_dataset

    if "MNIST" in tgt_dataset:
        c, w_img, h_img = 1, 28, 28
        n_data_by_class_tgt = 5000
        n_projs = 500
    elif tgt_dataset == "CIFAR10":
        c, w_img, h_img = 3, 32, 32
        n_data_by_class_tgt = 5000
        n_projs = 500

    # Load dataset
    X_data_tgt, y_tgt, X_test, y_test = get_dataset(key_tgt, tgt_dataset,
                                                    n_data_by_class_tgt,
                                                    path_data)
    n_class, _, d = X_data_tgt.shape

    L_acc = np.zeros((args.ntry, args.neval))
    L_acc_baseline = np.zeros((args.ntry, args.neval))

    for i in range(args.ntry):
        master_key, key_x0, key_wgd = jax.random.split(master_key, 3)

        if args.init_true:
            inds = jax.random.randint(key_x0, (n_class, args.n_data_by_class),
                                      0, n_data_by_class_tgt)

            x0 = np.zeros((n_class, args.n_data_by_class, d))
            y0 = np.zeros((n_class, args.n_data_by_class))
            for k in range(n_class):
                for j in range(args.n_data_by_class):
                    x0[k, j] = X_data_tgt[k, inds[k, j]]
                    y0[k, j] = k
        else:
            x0 = jax.random.normal(key_x0, (n_class, args.n_data_by_class, d))

        lr = args.lr_wgd
        m = args.m_wgd

        # Get synthetic images
        if args.method == "dm":
            # No embedding and no augmentation
            @jax.jit
            def target_value_and_grad(x, y, rng):
                return target_value_and_grad_dm_full_class_ambient(
                    x, y, rng, args.batch_size_wgd
                    )

            L_loss, xk = wasserstein_gradient_descent(
                x0, X_data_tgt, target_value_and_grad, key_wgd,
                n_epochs=args.n_epochs_wgd, lr=lr, m=m
                )

        elif args.method == "dm_emb":
            # No augmentation
            @jax.jit
            def target_value_and_grad(x, y, rng):
                return target_value_and_grad_dm_full_class_only_emb(
                    x, y, rng, args.batch_size_wgd, c=c, w_img=w_img,
                    h_img=h_img
                    )

            L_loss, xk = wasserstein_gradient_descent(
                x0, X_data_tgt, target_value_and_grad, key_wgd,
                n_epochs=args.n_epochs_wgd, lr=lr, m=m
                )

        elif args.method == "dm_aug":
            # No embedding
            if tgt_dataset == "MNIST":
                @jax.jit
                def target_value_and_grad(x, y, rng):
                    return target_value_and_grad_dm_full_class_only_aug_mnist(
                        x, y, rng, args.batch_size_wgd, c=c, w_img=w_img,
                        h_img=h_img
                        )
            else:
                @jax.jit
                def target_value_and_grad(x, y, rng):
                    return target_value_and_grad_dm_full_class_only_aug(
                        x, y, rng, args.batch_size_wgd, c=c, w_img=w_img,
                        h_img=h_img
                        )

            L_loss, xk = wasserstein_gradient_descent(
                x0, X_data_tgt, target_value_and_grad, key_wgd,
                n_epochs=args.n_epochs_wgd, lr=lr, m=m
                )

        elif args.method == "dm_aug_emb":
            # Embedding and Augmentation
            if tgt_dataset == "MNIST":
                @jax.jit
                def target_value_and_grad(x, y, rng):
                    return target_value_and_grad_dm_full_class_emb_mnist(
                        x, y, rng, args.batch_size_wgd, c=c, w_img=w_img,
                        h_img=h_img
                        )
            else:
                @jax.jit
                def target_value_and_grad(x, y, rng):
                    return target_value_and_grad_dm_full_class_emb(
                        x, y, rng, args.batch_size_wgd, c=c, w_img=w_img,
                        h_img=h_img
                        )

            L_loss, xk = wasserstein_gradient_descent(
                x0, X_data_tgt, target_value_and_grad, key_wgd,
                n_epochs=args.n_epochs_wgd, lr=lr, m=m
                )

        elif args.method == "mmdsw":
            # No embedding and no augmentation
            @jax.jit
            def target_value_and_grad(x, y, rng):
                return target_value_and_grad_riesz(
                    x, y, rng, n_sample_batch=args.batch_size_wgd,
                    n_projs=n_projs
                )
            L_loss, xk = wasserstein_gradient_descent(
                x0, X_data_tgt, target_value_and_grad, key_wgd,
                n_epochs=args.n_epochs_wgd, lr=lr, m=m
                )

        elif args.method == "mmdsw_emb":
            # No augmentation
            @jax.jit
            def target_value_and_grad(x, y, rng):
                return target_value_and_grad_riesz_dataset_distillation(
                    x, y, rng, args.batch_size_wgd, c=c, w_img=w_img,
                    h_img=h_img,
                    n_projs=n_projs
                )
            L_loss, xk = wasserstein_gradient_descent(
                x0, X_data_tgt, target_value_and_grad, key_wgd,
                n_epochs=args.n_epochs_wgd, lr=lr, m=m
                )

        elif args.method == "mmdsw_aug":
            # No embedding
            if tgt_dataset == "MNIST":
                @jax.jit
                def target_value_and_grad(x, y, rng):
                    return target_value_and_grad_riesz_dataset_distillation_no_emb_aug_mnist(
                        x, y, rng, args.batch_size_wgd, c=c, w_img=w_img,
                        h_img=h_img, n_projs=n_projs
                    )
            else:
                @jax.jit
                def target_value_and_grad(x, y, rng):
                    return target_value_and_grad_riesz_dataset_distillation_no_emb_aug_full(
                        x, y, rng, args.batch_size_wgd, c=c, w_img=w_img,
                        h_img=h_img, n_projs=n_projs
                    )

            L_loss, xk = wasserstein_gradient_descent(
                x0, X_data_tgt, target_value_and_grad, key_wgd,
                n_epochs=args.n_epochs_wgd, lr=lr, m=m
                )

        elif args.method == "mmdsw_aug_emb":
            # Augmentation + Embedding
            if tgt_dataset == "MNIST":
                @jax.jit
                def target_value_and_grad(x, y, rng):
                    return target_value_and_grad_riesz_dataset_distillation_aug_mnist(
                        x, y, rng, args.batch_size_wgd, c=c, w_img=w_img,
                        h_img=h_img, n_projs=n_projs
                    )
            else:
                @jax.jit
                def target_value_and_grad(x, y, rng):
                    return target_value_and_grad_riesz_dataset_distillation_aug_full(
                        x, y, rng, args.batch_size_wgd, c=c, w_img=w_img,
                        h_img=h_img, n_projs=n_projs
                    )

            L_loss, xk = wasserstein_gradient_descent(
                x0, X_data_tgt, target_value_and_grad, key_wgd,
                n_epochs=args.n_epochs_wgd, lr=lr, m=m
                )

        yk = get_labels(X_data_tgt, y_tgt, xk)

        # Evaluate synthetic images by training neural networks on it
        for j in range(args.neval):
            master_key, key_train_nn, key_b = jax.random.split(master_key, 3)
            model = train_nn(key_train_nn, np.array(xk), np.array(yk),
                             n_epochs=1000, c=c, w_img=w_img, h_img=h_img)
            test_accuracy = eval_nn(model, X_test, y_test,
                                    c=c, w_img=w_img, h_img=h_img)
            L_acc[i, j] = test_accuracy

            if args.init_true:
                model_baseline = train_nn(key_b, np.array(x0), np.array(y0),
                                          n_epochs=1000, c=c, w_img=w_img,
                                          h_img=h_img)
                test_accuracy = eval_nn(model_baseline, X_test, y_test,
                                        c=c, w_img=w_img, h_img=h_img)
                L_acc_baseline[i, j] = test_accuracy

    path_acc = "/accuracy_" + tgt_dataset + "_" + args.method +\
               "_n" + str(args.n_data_by_class) +\
               "_lr" + str(args.lr_wgd) + "_m" +\
               str(args.m_wgd) + "_nepochs" + str(args.n_epochs_wgd)

    np.savetxt(args.path_results + path_acc, L_acc, delimiter=",")

    path_imgs = "/synth_images_" + tgt_dataset + "_" + args.method +\
                "_n" + str(args.n_data_by_class) + "_lr" +\
                str(args.lr_wgd) + "_m" + str(args.m_wgd) +\
                "_nepochs" + str(args.n_epochs_wgd)

    np.save(args.path_results + path_imgs, xk)

    path_loss = "/loss_" + tgt_dataset + "_" + args.method +\
                "_n" + str(args.n_data_by_class) + "_lr" +\
                str(args.lr_wgd) + "_m" + str(args.m_wgd) +\
                "_nepochs" + str(args.n_epochs_wgd)

    np.savetxt(args.path_results + path_loss, L_loss, delimiter=",")

    if args.init_true:
        path_acc_baseline = "/accuracy_" + tgt_dataset + "_baseline" +\
            "_n" + str(args.n_data_by_class)

        np.savetxt(args.path_results + path_acc_baseline, L_acc_baseline,
                   delimiter=",")
