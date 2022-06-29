import torch
import torch.nn.functional as F


def conv2d(x, k, b, method='naive'):
    """
    Convolution of single instance and single input and output channel
    :param x:  (H, W) PyTorch Tensor
    :param k:  (Hk, Wk) PyTorch Tensor
    :param b:  (1,) PyTorch tensor or scalar
    :param method: Which method do we use to implement it. Valid choices include
                   'naive', 'torch', 'pytorch', 'im2col', 'winograd', and 'fft'
    :return:
        Output tensor should have shape (H_out, W_out)
    """
    method = method.lower()
    if method == 'naive':
        return naive(x, k, b)
    elif method in ['torch', 'pytorch']:
        return pytorch(x, k, b)
    elif method == 'im2col':
        return im2col(x, k, b)
    elif method == 'winograd':
        return winograd(x, k, b)
    elif method == 'fft':
        return fft(x, k, b)
    else:
        raise ValueError("Invalid [method] value: %s" % method)


def naive(x, k, b):
    """ Sliding window solution. """
    output_shape_0 = x.shape[0] - k.shape[0] + 1
    output_shape_1 = x.shape[1] - k.shape[1] + 1
    result = torch.zeros(output_shape_0, output_shape_1)
    for row in range(output_shape_0):
        for col in range(output_shape_1):
            window = x[row: row + k.shape[0], col: col + k.shape[1]]
            result[row, col] = torch.sum(torch.multiply(window, k))
    return result + b


def pytorch(x, k, b):
    """ PyTorch solution. """
    return F.conv2d(
        x.unsqueeze(0).unsqueeze(0),  # (1, 1, H, W)
        k.unsqueeze(0).unsqueeze(0),  # (1, 1, Hk, Wk)
        b   # (1, )
    ).squeeze(0).squeeze(0)  # (H_out, W_out)


def im2col(x, k, b):
    little0, little1 = k.shape
    big0, big1 = x.shape
    stack = []
    for i0 in range(big0 - little0 + 1):
        for i1 in range(big1 - little1 + 1):
            stack.append(x[i0:i0+little0, i1:i1+little1].flatten())
    stacked = torch.stack(stack).T
    a = (k.reshape(1, -1) @ stacked)
    return a.reshape(big0 - little0 + 1, big1 - little1 + 1) + b


def winograd(x, k, b):
    little0, little1 = k.shape
    big0, big1 = x.shape
    pad_x = torch.nn.functional.pad(x, (0,1,0,1))
    input_thingy = torch.tensor([[1,0,-1,0],[0,1,1,0],[0,-1,1,0],[0,1,0,-1]], dtype=x.dtype)
    kernel_thingy = torch.tensor([[1,0,0],[.5,.5,.5],[.5,-.5,.5],[0,0,1]], dtype=x.dtype)
    output_thingy = torch.tensor([[1,1,1,0],[0,1,-1,-1]], dtype=x.dtype)
    kk = kernel_thingy @ k @ kernel_thingy.T
    output = torch.zeros(big0 - little0 + 2, big1 - little1 + 2)
    for ii0, i0 in enumerate(range(0, big0 - little0 + 2, 2)):
        for ii1, i1 in enumerate(range(0, big1 - little1 + 2, 2)):
            output[i0:i0+2,i1:i1+2] += output_thingy @ ((input_thingy @ pad_x[i0:i0+4, i1:i1+4] @ input_thingy.T) * kk) @ output_thingy.T
    return output[:-1, :-1] + b

def fft(x, k, b):
    little0, little1 = k.shape
    big0, big1 = x.shape
    #pad_x = torch.nn.functional.pad(x, ((little0 - 1) // 2, (little0 - 1) // 2, (little1 - 1) // 2, (little1 - 1) // 2))
    pad_k = torch.zeros((big0, big1), dtype=x.dtype)
    g_y, g_x = torch.meshgrid(torch.arange(little0), torch.arange(little1))
    g_new_y = (g_y.flip(0) - little0 // 2) % pad_k.size(0)
    g_new_x = (g_x.flip(1) - little1 // 2) % pad_k.size(1)
    pad_k[g_new_y, g_new_x] = k[g_y, g_x]
    #print(pad_k)
    k_fft = torch.fft.fft2(pad_k)
    x_fft = torch.fft.fft2(x)
    result = torch.real(torch.fft.ifft2(k_fft * x_fft))
    if little0 == 1:
        return result + b
    return result[(little0 - 1) // 2:-(little0 - 1) // 2,(little1 - 1) // 2:-(little1 - 1) // 2] + b