import torch
import numpy as np

from sklearn.decomposition import PCA
from utils.evaluate import mean_average_precision, pr_curve


def train(
    train_data,
    query_data,
    query_targets,
    retrieval_data,
    retrieval_targets,
    code_length,
    device,
    topk,
    ):
    """
    Training model.

    Args
        train_data(torch.Tensor): Training data.
        query_data(torch.Tensor): Query data.
        query_targets(torch.Tensor): Query targets.
        retrieval_data(torch.Tensor): Retrieval data.
        retrieval_targets(torch.Tensor): Retrieval targets.
        code_length(int): Hash code length.
        device(torch.device): GPU or CPU.
        topk(int): Calculate top k data points map.

    Returns
        checkpoint(dict): Checkpoint.
    """
    # PCA
    pca = PCA(n_components=code_length)
    X = pca.fit_transform(train_data.numpy())

    # Fit uniform distribution
    eps = np.finfo(float).eps
    mn = X.min(0) - eps
    mx = X.max(0) + eps

    # Enumerate eigenfunctions
    R = mx - mn
    max_mode = np.ceil((code_length + 1) * R / R.max()).astype(np.int)
    n_modes = max_mode.sum() - len(max_mode) + 1
    modes = np.ones([n_modes, code_length])
    m = 0
    for i in range(code_length):
        modes[m + 1: m + max_mode[i], i] = np.arange(1, max_mode[i]) + 1
        m = m + max_mode[i] - 1

    modes -= 1
    omega0 = np.pi / R;
    omegas = modes * omega0.reshape(1, -1).repeat(n_modes, 0)
    eig_val = -(omegas ** 2).sum(1)
    ii = (-eig_val).argsort()
    modes = modes[ii[1: code_length + 1], :]

    # Evaluate
    # Generate query code and retrieval code
    query_code = generate_code(query_data.cpu(), code_length, pca, mn, R, modes).to(device)
    retrieval_code = generate_code(retrieval_data.cpu(), code_length, pca, mn, R, modes).to(device)
    query_targets = query_targets.to(device)
    retrieval_targets = retrieval_targets.to(device)

    # Compute map
    mAP = mean_average_precision(
        query_code,
        retrieval_code,
        query_targets,
        retrieval_targets,
        device,
        topk,
    )

    # P-R curve
    P, Recall = pr_curve(
        query_code,
        retrieval_code,
        query_targets,
        retrieval_targets,
        device,
    )

    # Save checkpoint
    checkpoint = {
        'qB': query_code,
        'rB': retrieval_code,
        'qL': query_targets,
        'rL': retrieval_targets,
        'P': P,
        'R': Recall,
        'map': mAP,
    }

    return checkpoint


def generate_code(data, code_length, pca, mn, R, modes):
    """
    Generate hashing code.

    Args
        data(torch.Tensor): Data.
        code_length(int): Hashing code length.
        R(torch.Tensor): Rotration matrix.
        pca(callable): PCA function.

    Returns
        pca_data(torch.Tensor): PCA data.
    """
    data = pca.transform(data.numpy()) - mn.reshape(1, -1)
    omega0 = np.pi / R
    omegas = modes * omega0.reshape(1, -1)
    U = np.zeros((len(data), code_length))
    for i in range(code_length):
        omegai = omegas[i, :]
        ys = np.sin(data * omegai + np.pi / 2)
        yi = np.prod(ys, 1)
        U[:, i] = yi
    
    return torch.from_numpy(np.sign(U))

