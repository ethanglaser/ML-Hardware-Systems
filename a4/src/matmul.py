import torch
import torch.nn.functional as F


def matmul(A, B, method='naive', **kwargs):
    """
    Multiply two matrices.
    :param A: (N, M) torch tensor.
    :param B: (M, K) torch tensor.
    :param method:
    :return:
        Output matrix with shape (N, K)
    """
    method = method.lower()
    if method in ['naive', 'pytorch', 'torch']:
        return naive(A, B)
    elif method == 'svd':
        return svd(A, B, **kwargs)
    elif method in ['log', 'logmatmul']:
        return logmatmul(A, B, **kwargs)
    else:
        raise ValueError("Invalid [method] value: %s" % method)


def naive(A, B, **kwargs):
    return A @ B


def svd(A, B, rank_A=None, rank_B=None):
    """
    Apply low-rank approximation (SVD) to both matrix A and B with rank rank_A
    and rank_B respectively.
    :param A: (N, M) pytorch tensor
    :param B: (M, K) pytorch tensor
    :param rank_A: None or int. None means use original A matrix.
    :param rank_B: None or int. None means use original B matrix.
    :return: a (N, K) pytorch tensor
    """
    if rank_A:
        U_a, S_a, V_a = torch.svd(A)
        A = U_a[:, :rank_A] @ (S_a[:rank_A].reshape(-1, 1) * V_a[:, :rank_A].T)
    if rank_B:
        U_b, S_b, V_b = torch.svd(B)
        B = U_b[:, :rank_B] @ (S_b[:rank_B].reshape(-1, 1) * V_b[:, :rank_B].T)
    return A @ B


def logmatmul(A, B, **kwargs):
    sign_a = torch.sign(A)
    sign_b = torch.sign(B)
    logpos_a = torch.log2(torch.abs(A))
    logpos_b = torch.log2(torch.abs(B))
    output = torch.zeros((A.shape[0], B.shape[1]))
    for x in range(A.shape[0]):
        for y in range(B.shape[1]):
            for z in range(A.shape[1]):
                if sign_a[x][z] + sign_b[z][y]:
                    output[x][y] += 2 ** (logpos_a[x][z] + logpos_b[z][y])
                else:
                    output[x][y] -= 2 ** (logpos_a[x][z] + logpos_b[z][y])
    return output
