'''
demo_vector_quantization.py

'''

import numpy as np
import matplotlib.pyplot as plt
from vector_quantization_utils import KmeansQuantizer, ApplyQuantization
import math

# simulated data
N_levels_old = 256; # number of levels before quantization
R_old = np.log2(N_levels_old); # rate before quantization [bits / symbol]
N_symbols = 1200; # number of points
t = np.linspace(0, 5, N_symbols); # time axis
w = 2; # angular frequency
signal_input = np.sin(w * t); # simulated signal
signal_input_max = np.abs(signal_input).max();
signal_input = signal_input / signal_input_max;

# choosing the quantization parameters
mu = 0; # pdf mean
sigma = 1; # pdf standard deviation
func = 'norm'; # sample pdf ('uniform', 'expon', 'gamma', 'rayleigh', 'norm')
N_iter = 10; # the number of random initializations for N_levels recon levels
N_samples = 120000; # the number of training samples drawn from the func pdf
R = 1; # the rate of quantization [bits / symbol]
N_levels = 256; # the number of quantization levels
N_dim = math.log2(N_levels) / R; # the number of grouped scalars (the dimentionality of quantizer)
tol = 10e-6; # tolerance 

if N_dim % 2 != 0 and N_dim % 2 != 1.0:
    raise ValueError(f'N_dim should be an integral number! Given value N_dim = {N_dim}\nAborting..');
N_dim = int(N_dim);
 
recon_levels, recon_error = KmeansQuantizer(func, N_iter, N_samples, N_dim, N_levels, mu, sigma, tol);
signal_output, quant_error, SQNR = ApplyQuantization(recon_levels, signal_input);

signal_input *= signal_input_max;
signal_output *= signal_input_max;
quant_error *= signal_input_max;

print(f'Size of the original signal: {R_old * N_symbols / 1000} [bytes]')
print(f'Size of the quantized signal: {R * N_symbols / 1000} [bytes]')
print(f'Compression ratio: {R_old / R : 0.5f}\n');

fig, ((ax1, ax2)) = plt.subplots(nrows=1, ncols=2);
ax1.plot(t, signal_input, '-b', label = 'Signal original');
ax1.plot(t, signal_output, '-r', label = 'Signal quantized');
ax1.plot(t, quant_error, '-k', label = 'Quantization error')
ax1.set_xlabel('Time [a.u.]')
ax1.set_ylabel('Signal [a.u.]')
ax1.set_title(f'Q Levels = {N_levels}\nQ group size = {N_dim}\n Q rate = {R} bits / symbol\nSQNR = {SQNR:0.4f}')
ax1.legend()

ax2.plot(signal_input, signal_output, '-k');
ax2.set_xlabel('Signal original [a.u.]')
ax2.set_ylabel('Signal quantized [a.u.]')

plt.tight_layout()