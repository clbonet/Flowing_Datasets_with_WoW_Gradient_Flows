import torch
import jax
import optax
import sys

sys.path.append("../")
from lib.classif_nn import ConvNet, train, Dataset


def train_nn(rng, X_train, y_train, batch_size=64, lr=3e-4, n_epochs=10,
             c=1, w_img=28, h_img=28):
    X_train = X_train.reshape(-1, c, w_img, h_img)
    y_train = y_train.reshape(-1,).astype(int)

    train_dataset = Dataset(X_train, y_train)

    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    master_key, key_cnn = jax.random.split(rng)
    model = ConvNet(key_cnn, channel=c, im_size=(w_img, h_img))

    optim = optax.adamw(lr)

    model, _, _ = train(model, trainloader, optim, n_epochs)
    return model
