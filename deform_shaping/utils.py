import torch


def polar_decompose(A):
    """ Polar decomposition for matrix."""
    assert A.shape[-1] == A.shape[-2]

    U, S, Vh = torch.linalg.svd(A)
    P = Vh.transpose(-2, -1) @ torch.diag_embed(S) @ Vh
    U = U @ Vh
    return U, P
