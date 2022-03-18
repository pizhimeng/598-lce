from curses import window
from math import sin, cos, exp, pi
import torch

# def sinc(x):
#     if x == 0:
#         return 1
#     else:
#         return sin(pi*x)/(pi*x)

# sinc = torch.sinc

def hann_window(trans_width:float) -> torch.tensor:
    trans_width *= pi
    size = int(6.65*pi/trans_width)
    if size % 2 == 0:
        size = size + 1

    window = torch.arange(size, dtype=float)
    window = torch.mul(window, pi)
    window = torch.div(window, size-1)
    window = torch.sin(window)
    window = torch.square(window)
    return window

def hamm_window(trans_width:float) -> torch.tensor:
    trans_width *= pi
    size = int(6.22*pi/trans_width)
    if size % 2 == 0:
        size = size + 1
    
    window = torch.arange(size, dtype=float)
    window = torch.mul(window, pi)
    window = torch.div(window, size-1)
    window = torch.cos(window)
    window = torch.mul(window, 0.46)
    window = torch.sub(0.54, window)
    return window

def gauss_window(trans_width:float) -> torch.tensor:
    trans_width *= pi
    size = int(8.52*pi/trans_width)
    if size % 2 == 0:
        size = size + 1

    window = torch.arange( -int((size-1)/2), int((size-1)/2) + 1, dtype=float)
    stdev = (size-1)/5
    window = torch.div(window, stdev)
    window = torch.square(window)
    window = torch.mul(window, -0.5)
    window = torch.exp(window)
    return window

def black_window(trans_width:float) -> torch.tensor:
    trans_width *= pi
    size = int(11.1*pi/trans_width)
    if size % 2 == 0:
        size = size + 1
    
    window1 = torch.arange( -int((size-1)/2), int((size-1)/2) + 1, dtype=float)
    window1 = torch.mul(window1, 2*pi/(size-1))
    window1 = torch.cos(window1)
    window1 = torch.mul(window1, 0.5)
    window1 = torch.add(window1, 0.42)

    window2 = torch.arange( -int((size-1)/2), int((size-1)/2) + 1, dtype=float)
    window2 = torch.mul(window2, 4*pi/(size-1))
    window2 = torch.cos(window2)
    window2 = torch.mul(window2, 0.08)

    window = torch.add(window1, window2)
    return window

def build_window(name:str, trans_width:float) -> torch.tensor:
    #name is a string, telling the program which window to use. trans_width is the width of the transition band in rad/s

    if name == 'hanning':
        window = hann_window(trans_width)

    elif name == 'hamming':
        window = hamm_window(trans_width)

    elif name == 'gaussian':
        window = gauss_window(trans_width)

    elif name == 'blackman':
        window = black_window(trans_width)

    else:
        print ('Invalid window name')
        
    return window

def build_filter(Wc:float, window:torch.tensor) -> torch.tensor:
    size = window.shape[0]
    x = torch.arange( -int((size-1)/2), int((size-1)/2) + 1, dtype=float)
    kernel = torch.mul(x, Wc)
    kernel = torch.sinc(kernel)
    kernel = torch.mul(kernel, Wc)
    kernel = torch.mul(kernel, window)
    return kernel


# def conv_old(f,g):      #f is kernel, g is signal
#     nf = len(f)
#     ng = len(g)
#     n = nf + ng - 1
#     out = []
#     for i in range(n):
#         result = 0.0
#         for j in range(nf):
#             if i - j < 0 or i - j >= ng:
#                 result += 0.0
#             else:
#                 result += f[j] * g[i - j]
#         out.append(result)
#     return out[int(nf/2) : int(nf/2+ng)]

def conv(kernel:torch.tensor, signal:torch.tensor) -> torch.tensor:
    nf = len(kernel)
    ng = len(signal)

    n = nf + ng - 1

    rst = torch.zeros(n, dtype=float)

    for i in range(n):
        for j in range(nf):
            if i - j < 0 or i - j >= ng:
                rst[i] += 0.0
            else:
                rst[i] += kernel[j] * signal[i-j]

    return rst[int(nf/2) : int(nf/2+ng)] 

# signal = []

# with open('signal.txt') as f:
#     for line in f:
#         l = line.split()
#         for num in l:
#             signal.append(float(num))

# name = raw_input('Please enter the name of window: ')
# trans_width = float(raw_input('Please enter the desired width of the transition band: '))
# Wc = float(raw_input('Please enter the desired cut-off frequency: '))

# window = build_window(name,trans_width)
# kernel = build_filter(Wc,window)
# filtered_signal = conv(kernel,signal)

# outfile = open('D:\Documents\MATLAB\SKA project Perth\output.txt','w')
# for data in filtered_signal:
#     outfile.write('%f ' % data)
# outfile.close()

