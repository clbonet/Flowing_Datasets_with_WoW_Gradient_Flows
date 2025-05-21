import jax
import torch

import jax.numpy as jnp
import equinox as eqx

from jaxtyping import Array, Float, Int, PyTree  # https://github.com/google/jaxtyping
from tqdm import trange


class ConvNet(eqx.Module):
    """
        Architecture from https://github.com/VICO-UoE/DatasetCondensation/blob/master/networks.py

        Used for Dataset Distillation experiment on MNIST, FMNIST, CIFAR10
    """
    layers_cnn: list
    classifier: eqx.nn.Linear

    def __init__(self, key, channel=1, net_width=128, net_depth=3, net_norm="instancenorm", net_pooling="avgpooling", im_size=(28,28), num_classes=10):
        master_key, key_cnn, key_classifier = jax.random.split(key, 3)

        # Standard CNN setup: convolutional layer, followed by flattening,
        # with a small MLP on top.
        self.layers_cnn, shape_feat = self._make_layers(key_cnn, channel, net_width, net_depth, net_norm, net_pooling, im_size)
        num_feat = shape_feat[0]*shape_feat[1]*shape_feat[2]
        self.classifier = eqx.nn.Linear(num_feat, num_classes, key=key_classifier)

    def embed(self, x):
        for layer in self.layers_cnn:
            x = layer(x)
        return x    

    def __call__(self, x):
        x = self.embed(x)
        return jax.nn.log_softmax(self.classifier(jnp.ravel(x)))

    def _make_layers(self, key, channel, net_width, net_depth, net_norm, net_pooling, im_size):
        layers = []

        in_channels = channel
        if im_size[0] == 28:
            im_size = (32, 32)

        shape_feat = [in_channels, im_size[0], im_size[1]]

        for d in range(net_depth):
            master_key, key = jax.random.split(key, 2)

            layers += [eqx.nn.Conv2d(in_channels, net_width, kernel_size=3, padding=3 if channel == 1 and d == 0 else 1, key=key)]
            shape_feat[0] = net_width

            if net_norm != 'none':
                layers += [eqx.nn.GroupNorm(shape_feat[0], shape_feat[0], channelwise_affine=True)]
            layers += [jax.nn.relu]

            in_channels = net_width
            if net_pooling != 'none':
                layers += [eqx.nn.AvgPool2d(kernel_size=2, stride=2)]
                shape_feat[1] //= 2
                shape_feat[2] //= 2

        return layers, shape_feat


class LeNet5(eqx.Module):
    """
        LeNET5 CNN used for Transfer Learning on *NIST datasets
    """
    layers: list

    def __init__(self, key):
        key1, key2, key3, key4, key5 = jax.random.split(key, 5)
        self.layers = [
            eqx.nn.Conv2d(1, 6, kernel_size=1, key=key1),
            jax.nn.relu,
            eqx.nn.AvgPool2d(kernel_size=2, stride=2),
            eqx.nn.Conv2d(6, 16, kernel_size=5, key=key2),
            jax.nn.relu,
            eqx.nn.AvgPool2d(kernel_size=2, stride=2),
            eqx.nn.Conv2d(16, 120, kernel_size=5, key=key3),
            jax.nn.relu,
            jnp.ravel,
            eqx.nn.Linear(120, 84, key=key4),
            jax.nn.relu,
            eqx.nn.Linear(84, 10, key=key5),
            jax.nn.log_softmax,
        ]

    def __call__(self, x: Float[Array, "1 28 28"]) -> Float[Array, "10"]:
        for layer in self.layers:
            x = layer(x)
        return x
    
    
class CNN(eqx.Module):
    """
        CNN from https://docs.kidger.site/equinox/examples/mnist/, 
        used for Domain Adaptation sanity checks.
    """
    layers: list

    def __init__(self, key):
        key1, key2, key3, key4 = jax.random.split(key, 4)
        # Standard CNN setup: convolutional layer, followed by flattening,
        # with a small MLP on top.
        self.layers = [
            eqx.nn.Conv2d(1, 3, kernel_size=4, key=key1),
            eqx.nn.MaxPool2d(kernel_size=2),
            jax.nn.relu,
            jnp.ravel,
            eqx.nn.Linear(1728, 512, key=key2),
            jax.nn.sigmoid,
            eqx.nn.Linear(512, 64, key=key3),
            jax.nn.relu,
            eqx.nn.Linear(64, 10, key=key4),
            jax.nn.log_softmax,
        ]

    def __call__(self, x: Float[Array, "1 28 28"]) -> Float[Array, "10"]:
        for layer in self.layers:
            x = layer(x)
        return x


@eqx.filter_jit
def loss(model, x, y):
    # model handle image (c, d, d)
    # Therefore, we have to use jax.vmap, which in this case maps our model over the
    # leading (batch) axis.
    pred_y = jax.vmap(model)(x)
    return cross_entropy(y, pred_y)


def cross_entropy(
    y: Int[Array, " batch"], pred_y: Float[Array, "batch 10"]
) -> Float[Array, ""]:
    # y are the true targets, and should be integers 0-9.
    # pred_y are the log-softmax'd predictions.
    pred_y = jnp.take_along_axis(pred_y, jnp.expand_dims(y, 1), axis=1)
    return -jnp.mean(pred_y)


def train(model, trainloader, optimizer, n_epochs):
    pbar = trange(n_epochs)

    L_loss = []
    L_acc = []

    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def make_step(model, opt_state, x, y):
        loss_value, grads = eqx.filter_value_and_grad(loss)(model, x, y)
        updates, opt_state = optimizer.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        acc = jnp.sum(jnp.argmax(jax.vmap(model)(x), axis=-1) == y)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_value, acc

    for e in pbar:
        loss_avg, acc_avg, num_exp = 0, 0, 0

        for i, (x, y) in enumerate(trainloader):
            x, y = x.numpy(), y.numpy()
            model, opt_state, train_loss, acc = make_step(model, opt_state, x, y)

            loss_avg += train_loss * y.shape[0]
            acc_avg += acc
            num_exp += y.shape[0]

        loss_avg /= num_exp
        acc_avg /= num_exp

        pbar.set_postfix({"loss": loss_avg, "acc": acc_avg})

        L_loss.append(loss_avg)
        L_acc.append(acc_avg)

    return model, L_loss, L_acc


@eqx.filter_jit
def compute_accuracy(model, x, y):
    """This function takes as input the current model
    and computes the average accuracy on a batch.
    """
    pred_y = jax.vmap(model)(x)
    pred_y = jnp.argmax(pred_y, axis=1)
    return jnp.mean(y == pred_y)


def evaluate(model: eqx.Module, testloader: torch.utils.data.DataLoader):
    """This function evaluates the model on the test dataset,
    computing both the average loss and the average accuracy.
    """
    avg_loss = 0
    avg_acc = 0
    for x, y in testloader:
        x = x.numpy()
        y = y.numpy()
        # Note that all the JAX operations happen inside `loss` and `compute_accuracy`,
        # and both have JIT wrappers, so this is fast.
        avg_loss += loss(model, x, y)
        avg_acc += compute_accuracy(model, x, y)
    return avg_loss / len(testloader), avg_acc / len(testloader)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        return sample, label
    

def eval_nn(model, X_test, y_test, batch_size=64, c=1, w_img=28, h_img=28):
    """
        Evaluate model over the test set, where X_test is of shape (nc, n, d)
    """
    X_test = X_test.reshape(-1, c, w_img, h_img)
    y_test = y_test.reshape(-1,).astype(int)

    test_dataset = Dataset(X_test, y_test)

    testloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True
    )

    test_loss, test_accuracy = evaluate(model, testloader)
    return test_accuracy
