import jax
import torch
import torchvision

import numpy as np
import jax.numpy as jnp
import torchvision.datasets as datasets


def get_dataset(rng, dataset, n_data_by_class, path_data="~/torch_datasets"):
    """
        params:
        - rng
        - dataset: string in "MNIST", "FMNIST", "KMNIST", "USPS"
        - n_data_by_class: int
        - path_data: str towards where data are

        returns:
        - X_data: ndarray of size (n_classes, n_data_by_class, d*d)
        - X_labels: ndarray of size (n_classes, n_data_by_class)
        - test_data: ndarray of size (n_test, d*d)
        - test_labels: ndarray of size (n_test,)
    """
    normalise_data = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            # torchvision.transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    if dataset == "MNIST":
        trainset = datasets.MNIST(root=path_data, train=True, download=True,
                                  transform=normalise_data)
        testset = datasets.MNIST(root=path_data, train=False, download=True,
                                 transform=normalise_data)

        n_classes = 10
        d = 28
        c = 1
    elif dataset == "FMNIST":
        trainset = datasets.FashionMNIST(root=path_data, train=True,
                                         download=True,
                                         transform=normalise_data)
        testset = datasets.FashionMNIST(root=path_data, train=False,
                                        download=True,
                                        transform=normalise_data)

        n_classes = 10
        d = 28
        c = 1
    elif dataset == "KMNIST":
        trainset = datasets.KMNIST(root=path_data, train=True,
                                   download=True,
                                   transform=normalise_data)
        testset = datasets.KMNIST(root=path_data, train=False,
                                  download=True,
                                  transform=normalise_data)

        n_classes = 10
        d = 28
        c = 1
    elif dataset == "USPS":
        normalise_data = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize((28, 28)),
            ]
        )

        trainset = datasets.USPS(root=path_data, train=True,
                                 download=True,
                                 transform=normalise_data)
        testset = datasets.USPS(root=path_data, train=False,
                                download=True,
                                transform=normalise_data)

        n_classes = 10
        d = 28
        c = 1

    elif dataset == 'CIFAR10':
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]

        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=mean, std=std)
            ]
        )

        trainset = datasets.CIFAR10(root=path_data, train=True,
                                    download=True, transform=transform)
        testset = datasets.CIFAR10(root=path_data, train=False,
                                   download=True,
                                   transform=transform)

        n_classes = 10
        d = 32
        c = 3

    elif dataset == "NormalizedMNIST":
        mean = [0.1307]
        std = [0.3081]
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=mean, std=std)])
        trainset = datasets.MNIST(root=path_data, train=True,
                                  download=True, transform=transform)
        testset = datasets.MNIST(root=path_data, train=False,
                                 download=True, transform=transform)

        n_classes = 10
        d = 28
        c = 1

    elif dataset == "NormalizedFMNIST":
        mean = [0.2861]
        std = [0.3530]
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=mean, std=std)])
        trainset = datasets.FashionMNIST(root=path_data, train=True,
                                         download=True, transform=transform)
        testset = datasets.FashionMNIST(root=path_data, train=False,
                                        download=True, transform=transform)

        n_classes = 10
        d = 28
        c = 1
        
    elif dataset == "NormalizedKMNIST":
        mean = [0.1307]
        std = [0.3081]
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=mean, std=std)])
        trainset = datasets.KMNIST(root=path_data, train=True,
                                  download=True, transform=transform)
        testset = datasets.KMNIST(root=path_data, train=False,
                                 download=True, transform=transform)

        n_classes = 10
        d = 28
        c = 1
        
    elif dataset == "NormalizedUSPS":
        mean = [0.1307]
        std = [0.3081]
        normalise_data = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize((28, 28)),
                torchvision.transforms.Normalize(mean=mean, std=std)
            ]
        )

        trainset = datasets.USPS(root=path_data, train=True,
                                 download=True,
                                 transform=normalise_data)
        testset = datasets.USPS(root=path_data, train=False,
                                download=True,
                                transform=normalise_data)

        n_classes = 10
        d = 28
        c = 1
        
    elif dataset == 'SVHN':
        mean = [0.4377, 0.4438, 0.4728]
        std = [0.1980, 0.2010, 0.1970]
        
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=mean, std=std)
            ]
        )
        trainset = datasets.SVHN(path_data, split='train', download=True, transform=transform)  # no augmentation
        testset = datasets.SVHN(path_data, split='test', download=True, transform=transform)

        n_classes = 10
        d = 32
        c = 3

    dataloader_train = torch.utils.data.DataLoader(dataset=trainset,
                                                   batch_size=len(trainset))
    train_data, train_labels = next(iter(dataloader_train))

    dataloader_test = torch.utils.data.DataLoader(dataset=testset,
                                                  batch_size=len(testset))
    test_data, test_labels = next(iter(dataloader_test))

    train_data = train_data.numpy().reshape(-1, c*d*d)
    train_labels = train_labels.numpy()
    test_data = test_data.numpy().reshape(-1, c*d*d)
    test_labels = test_labels.numpy()

    X_data = np.zeros((n_classes, n_data_by_class, c*d*d))
    X_labels = np.zeros((n_classes, n_data_by_class))

    master_key = rng
    for k in range(n_classes):
        master_key, key = jax.random.split(master_key)
        data_class_k = train_data[train_labels == k]
        X_data[k] = jax.random.choice(key, data_class_k, (n_data_by_class,),
                                      replace=False)
        X_labels[k] = k

    return X_data, X_labels, test_data, test_labels


def get_moments_from_dataset(X_data, X_labels):
    """
        Return 2 first moments from the dataset X_data for each class.

        Parameters
        ----------
        - X_data: array of shape (n, d)
        - X_labels: array of shape (n,)

        Returns
        -------
        - mus: array of shape (n_classes, reduced_dim)
        - covs: array of shape (n_classes, reduced_dim, reduced_dim)
        - mu_full: array of shape (n, reduced_dim)
        - cov_full: array of shape (n, reduced_dim, reduced_dim)
    """
    n, d = X_data.shape    
    n_classes = len(np.unique(X_labels))

    covs = np.zeros((n_classes, d, d))
    mus = np.zeros((n_classes, d))
    X_labels = X_labels.reshape(-1,).astype(int)

    for k in range(n_classes):
        mus[k] = jnp.mean(X_data[X_labels == k], axis=0)
        covs[k] = jnp.cov(X_data[X_labels == k].T)

    mu_full, cov_full = mus[X_labels], covs[X_labels]

    return mus, covs, mu_full, cov_full
