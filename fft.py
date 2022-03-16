import torch
import numpy as np

def dft(x:torch.tensor):
    N = x.shape[0]
    n = torch.arange(N, dtype=float)
    k = n.reshape((N,1))
    M = torch.exp(-2j * np.pi * k * n / N)
    return torch.matmul(M.type(torch.complex64), x.type(torch.complex64))


def fft(x:torch.tensor):
    N = x.shape[0]

    if N %2 > 0:
        raise ValueError("must be a power of 2")
    elif N <= 2:
        return dft(x)
    else:
        x_even = fft(x[::2])
        
        x_odd = fft(x[1::2])
        terms = torch.exp(-2j * np.pi * torch.arange(N, dtype=float) / N) 
        return torch.cat( [x_even + terms[:int(N/2)] * x_odd,
                            x_even + terms[int(N/2):] * x_odd])

