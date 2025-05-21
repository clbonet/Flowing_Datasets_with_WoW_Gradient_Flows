import jax
import torch
import optax
import sys
import argparse

import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
import equinox as eqx

from utils_classif_nn import pretrain_nn

sys.path.append("../")
from lib.mmd import target_value_and_grad_riesz
from lib.datasets import get_dataset
from lib.gd_images import wasserstein_gradient_descent_save
from lib.classif_nn import ConvNet, eval_nn
from lib.utils_labels import get_labels

parser = argparse.ArgumentParser()
parser.add_argument("--method", type=str, default="hierarchical_mmd", help="Method to use")
parser.add_argument("--ntry", type=int, default=1, help="Number of try")
parser.add_argument("--path_data", type=str, default="~/torch_datasets", help="Directory torch data")
parser.add_argument("--src_dataset", type=str, default="SVHN", help="Source dataset")
parser.add_argument("--tgt_dataset", type=str, default="CIFAR10", help="Target dataset")
parser.add_argument("--n_data_by_class", type=int, default=500, help="Number of data by class")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size training NN")
parser.add_argument("--pretrained", help="If true, use pretrain model", action="store_true")
parser.add_argument("--n_epochs_wgd", type=int, default=1000, help="Number of epochs for WGD")
parser.add_argument("--save_every", type=int, default=5000, help="Number of array to be saved")
args = parser.parse_args()


def flow_towards_tgt(rng, X_data_src, X_data_tgt, method):

    if method == "hierarchical_mmd":
        m = 0.9
        n_epochs = args.n_epochs_wgd
        lr = 1

        target_grad = lambda x, y, key: target_value_and_grad_riesz(x, y, key, n_sample_batch=None)

        L_loss, L_xk = wasserstein_gradient_descent_save(X_data_src, X_data_tgt, target_grad, rng,
                                                         lr=lr, m=m, n_epochs=n_epochs, save_interval=args.save_every)

    return L_xk

if __name__ == "__main__":
    n_try = args.ntry

    rng = jax.random.PRNGKey(0)
    master_key, key_trys, key_tgt, key_train_nn = jax.random.split(rng, num=4)

    # Training CNN on MNIST
    path_data = args.path_data
    n_data_by_class = args.n_data_by_class
    tgt_dataset = args.tgt_dataset

    X_data_tgt, y_tgt, X_test, y_test = get_dataset(key_tgt, tgt_dataset, n_data_by_class, path_data)

    if args.pretrained:
        model_og = ConvNet(rng, channel=3, im_size=(32, 32))
        model = eqx.tree_deserialise_leaves("./results/nn_pretrained_"+tgt_dataset+".eqx", model_og)
    else:
        model = pretrain_nn(key_train_nn, X_data_tgt, y_tgt, c=3, w_img=32, h_img=32)
        eqx.tree_serialise_leaves("./results/nn_pretrained_"+tgt_dataset+".eqx", model)

    test_accuracy = eval_nn(model, X_test, y_test, batch_size=args.batch_size, c=3, w_img=32, h_img=32)
    print("ACCURACY on " + args.tgt_dataset + ":", test_accuracy)

    keys = jax.random.split(key_trys, num=n_try)

    num_saves = (args.n_epochs_wgd // args.save_every) + 1

    L_acc_baseline = np.zeros((n_try,))
    L_acc_mmdsw = np.zeros((n_try, num_saves,))

    for k in range(n_try):
        print("TRY " + str(k))
        master_key, key_src, key_wgd = jax.random.split(keys[k], num=3)

        # Flowing src dataset to MNIST
        src_dataset = args.src_dataset
        X_data_src, y_src, _, _ = get_dataset(key_src, src_dataset, n_data_by_class, path_data)
        test_accuracy = eval_nn(model, X_data_src, y_src, batch_size=args.batch_size, c=3, w_img=32, h_img=32)
        print("ACCURACY on " + args.src_dataset + " before flow:", test_accuracy)
        L_acc_baseline[k] = test_accuracy

        L_xk = flow_towards_tgt(key_wgd, X_data_src, X_data_tgt, args.method)
        for i, xk in enumerate(L_xk):
            xk_labels = get_labels(X_data_tgt, y_tgt, xk)
            test_accuracy = eval_nn(model, np.array(xk), xk_labels, batch_size=args.batch_size, c=3, w_img=32, h_img=32)
            L_acc_mmdsw[k, i] = test_accuracy

    np.savetxt("./results/DA_evol_" + args.src_dataset + "_" + args.method, L_acc_mmdsw, delimiter=",")
