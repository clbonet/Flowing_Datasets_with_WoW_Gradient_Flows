# Code from https://docs.kidger.site/equinox/examples/mnist/

import torch
import jax
import optax
import sys

import jax.numpy as jnp
import equinox as eqx

from jaxtyping import Array, Float, Int, PyTree  # https://github.com/google/jaxtyping

sys.path.append("../")
from lib.classif_nn import CNN, loss, cross_entropy, compute_accuracy, evaluate, ConvNet, Dataset


def train(
    model,
    trainloader,
    testloader,
    optim,
    steps,
    print_every,
    key=None
):
    # Just like earlier: It only makes sense to train the arrays in our model,
    # so filter out everything else.
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    # Always wrap everything -- computing gradients, running the optimiser, updating
    # the model -- into a single JIT region. This ensures things run as fast as
    # possible.
    @eqx.filter_jit
    def make_step(model, opt_state, x, y):
        loss_value, grads = eqx.filter_value_and_grad(loss)(model, x, y)
        updates, opt_state = optim.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_value

    # Loop over our training dataset as many times as we need.
    def infinite_trainloader():
        while True:
            yield from trainloader

    for step, (x, y) in zip(range(steps), infinite_trainloader()):
        # PyTorch dataloaders give PyTorch tensors by default,
        # so convert them to NumPy arrays.
        x = x.numpy()
        y = y.numpy()

        model, opt_state, train_loss = make_step(model, opt_state, x, y)
        if testloader is not None and ((step % print_every) == 0 or (step == steps - 1)):
            test_loss, test_accuracy = evaluate(model, testloader)
            print(
                f"{step=}, train_loss={train_loss.item()}, "
                f"test_loss={test_loss.item()}, test_accuracy={test_accuracy.item()}"
            )

    # test_loss, test_accuracy = evaluate(model, testloader)
    return model


def pretrain_nn(rng, X_train, y_train, batch_size=64, lr=3e-4, n_epochs=5000, c=1, w_img=28, h_img=28):
    X_train = X_train.reshape(-1, c, w_img, h_img)
    y_train = y_train.reshape(-1,).astype(int)

    train_dataset = Dataset(X_train, y_train)

    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    master_key, key_cnn, key_train = jax.random.split(rng, 3)

    if c == 1:
        model = CNN(key_cnn)
    elif c == 3:
        model = ConvNet(key_cnn, channel=c, im_size=(w_img, h_img))

    optim = optax.adamw(lr)

    model = train(model, trainloader, None, optim, steps=n_epochs, print_every=n_epochs, key=key_train)
    return model
