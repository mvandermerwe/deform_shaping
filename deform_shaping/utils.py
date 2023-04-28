import torch


def polar_decompose(A):
    """ Polar decomposition for matrix."""
    assert A.shape[-1] == A.shape[-2]

    U, S, Vh = torch.linalg.svd(A)
    P = Vh.transpose(-2, -1) @ torch.diag_embed(S) @ Vh
    U = U @ Vh
    return U, P


def polar_decompose2d(A):
    assert A.shape[-1] == A.shape[-2] and A.shape[-1] == 2
    device = A.device
    dtype = A.dtype

    # U = torch.eye(2, device=device, dtype=dtype).unsqueeze(0).repeat(A.shape[0], 1, 1)
    # P = A.clone()

    detA = torch.det(A)
    adetA = torch.abs(detA)

    B = torch.zeros_like(A, device=device, dtype=dtype)
    B[:, 0, 0] = torch.where(detA >= 0, A[:, 0, 0] + A[:, 1, 1], A[:, 0, 0] - A[:, 1, 1])
    B[:, 0, 1] = torch.where(detA >= 0, A[:, 0, 1] - A[:, 1, 0], A[:, 0, 1] + A[:, 1, 0])
    B[:, 1, 0] = torch.where(detA >= 0, A[:, 1, 0] - A[:, 0, 1], A[:, 1, 0] + A[:, 0, 1])
    B[:, 1, 1] = torch.where(detA >= 0, A[:, 1, 1] + A[:, 0, 0], A[:, 1, 1] - A[:, 0, 0])

    adetB = torch.abs(torch.det(B))

    k = 1.0 / torch.sqrt(adetB)

    U = B * k[:, None, None]
    P = A.transpose(-2, -1) @ A + adetA[:, None, None] * k[:, None, None] * torch.eye(2, device=device,
                                                                                      dtype=dtype).unsqueeze(0).repeat(
        A.shape[0], 1, 1)

    return U, P


def polar_decompose2d_ms(A):
    assert A.shape[-1] == A.shape[-2] and A.shape[-1] == 2
    device = A.device
    dtype = A.dtype

    # U = torch.eye(2, device=device, dtype=dtype).unsqueeze(0).repeat(A.shape[0], 1, 1)
    # P = A.clone()

    detA = torch.det(A)
    adetA = torch.abs(detA)

    B = torch.zeros_like(A, device=device, dtype=dtype)
    B[:, :, 0, 0] = torch.where(detA >= 0, A[:, :, 0, 0] + A[:, :, 1, 1], A[:, :, 0, 0] - A[:, :, 1, 1])
    B[:, :, 0, 1] = torch.where(detA >= 0, A[:, :, 0, 1] - A[:, :, 1, 0], A[:, :, 0, 1] + A[:, :, 1, 0])
    B[:, :, 1, 0] = torch.where(detA >= 0, A[:, :, 1, 0] - A[:, :, 0, 1], A[:, :, 1, 0] + A[:, :, 0, 1])
    B[:, :, 1, 1] = torch.where(detA >= 0, A[:, :, 1, 1] + A[:, :, 0, 0], A[:, :, 1, 1] - A[:, :, 0, 0])

    adetB = torch.abs(torch.det(B))

    k = 1.0 / torch.sqrt(adetB)

    U = B * k[:, :, None, None]
    P = A.transpose(-2, -1) @ A + adetA[:, :, None, None] * k[:, :, None, None] * torch.eye(2, device=device,
                                                                                            dtype=dtype).unsqueeze(
        0).repeat(
        A.shape[0], A.shape[1], 1, 1)

    return U, P
