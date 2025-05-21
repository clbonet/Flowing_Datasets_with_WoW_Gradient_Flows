import torch
import jax
import optax
import sys

import numpy as np

from sklearn.neighbors import KNeighborsClassifier

sys.path.append("../")
from lib.classif_nn import LeNet5, train, Dataset


def train_nn(rng, X_train, y_train, batch_size=64, lr=3e-4, n_epochs=10, c=1,
             w_img=28, h_img=28):
    X_train = X_train.reshape(-1, c, w_img, h_img)
    y_train = y_train.reshape(-1,).astype(int)

    train_dataset = Dataset(X_train, y_train)

    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    master_key, key_cnn, key_train = jax.random.split(rng, 3)
    model = LeNet5(key_cnn)

    optim = optax.adamw(lr)

    model, _, _ = train(model, trainloader, optim, n_epochs)
    return model


def augment_dataset(X_particles, X_tgt, y_tgt, full_path=False, method="none"):
    """
        X_particles: ndarray of shape (n_traj, n_class, n_particles, d)
        X_tgt: ndarry of shape (n_class, n_particles, d)
        y_tgt: ndarray of shape (n_class, n_particles)
        full_path: If True, use the full trajectories
        method: str in ["none", "rescale", "clip", "normalize"]

        Returns
        -------
        X_augmented: ndarray of shape (n_class, n_particles, d)
        y_augmented: ndarray of shape (n_class, n_particles)
    """
    to_add = np.concatenate([X_particles[i] for i in range(len(X_particles))],
                            axis=1) if full_path else X_particles[-1]

    if method == "rescale":
        min_data = np.min(to_add, axis=-1, keepdims=True)
        to_add_centered = to_add - min_data
        max_min_diff = np.max(to_add, axis=-1, keepdims=True) - min_data
        to_add = to_add_centered / max_min_diff
    elif method == "clip":
        to_add = np.clip(to_add, 0, 1)

    n_class = X_particles.shape[1]
    X_augmented = np.concatenate([X_tgt, np.zeros(to_add.shape)], axis=1)
    y_augmented = np.concatenate([y_tgt, np.zeros((n_class, to_add.shape[1]))], axis=1)

    knn = KNeighborsClassifier(1)
    knn.fit(X_tgt.reshape(-1, 784), y_tgt.reshape(-1,).astype(int))

    for k in range(n_class):
        # Majority vote
        labels = knn.predict(X_particles[-1][k])
        unique_labels, counts = np.unique(labels, return_counts=True)
        l = unique_labels[np.argmax(counts)]
        X_augmented[l, X_tgt.shape[1]:, :] = to_add[k]
        y_augmented[l, X_tgt.shape[1]:] = l

    return np.array(X_augmented), np.array(y_augmented)
