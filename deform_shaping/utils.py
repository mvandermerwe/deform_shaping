import torch


def polar_decompose(A):
    """ Polar decomposition for matrix."""
    assert A.shape[-1] == A.shape[-2]

    U, S, Vh = torch.linalg.svd(A)
    P = Vh.T @ torch.diag(S) @ Vh
    U = U @ Vh
    return U, P
