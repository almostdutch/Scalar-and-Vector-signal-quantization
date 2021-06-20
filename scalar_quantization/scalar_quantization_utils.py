'''
scalar_quantization_utils.py

'''

from scipy.integrate import quad
from scipy.stats import uniform, expon, gamma, rayleigh, norm
import numpy as np
from numpy import linalg as LA

def ReconLevel(func, a, b, mu, sigma):
    # Optimum quantizer
    # Calculates and returns a tupil (recon_level, recon_error)
    # func = sample pdf ('uniform', 'expon', 'gamma', 'rayleigh', 'norm')
    # a = quantization region left boundary
    # b = quantization region right boundary
    # mu = pdf mean
    # sigma = pdf standard deviation
    
    if func == 'uniform':
        y1, _ = quad(lambda x : x * uniform.pdf(x, mu, sigma), a, b);
        y2, _ = quad(uniform.pdf, a, b, args = (mu, sigma)); 
        recon_level = y1 / y2;
        
        mse, _ = quad(lambda x : (x - recon_level) ** 2 * uniform.pdf(x, mu, sigma), a, b);
    elif func == 'expon':
        y1, _ = quad(lambda x : x * expon.pdf(x, mu, sigma), a, b);
        y2, _ = quad(expon.pdf, a, b, args = (mu, sigma)); 
        recon_level = y1 / y2;
        
        mse, _ = quad(lambda x : (x - recon_level) ** 2 * expon.pdf(x, mu, sigma), a, b);
    elif func == 'gamma':
        y1, _ = quad(lambda x : x * gamma.pdf(x, mu, sigma), a, b);
        y2, _ = quad(gamma.pdf, a, b, args = (mu, sigma)); 
        recon_level = y1 / y2;
        
        mse, _ = quad(lambda x : (x - recon_level) ** 2 * gamma.pdf(x, mu, sigma), a, b);
    elif func == 'rayleigh':
        y1, _ = quad(lambda x : x * rayleigh.pdf(x, mu, sigma), a, b);
        y2, _ = quad(rayleigh.pdf, a, b, args = (mu, sigma)); 
        recon_level = y1 / y2;
        
        mse, _ = quad(lambda x : (x - recon_level) ** 2 * rayleigh.pdf(x, mu, sigma), a, b);
    elif func == 'norm':
        y1, _ = quad(lambda x : x * norm.pdf(x, mu, sigma), a, b);
        y2, _ = quad(norm.pdf, a, b, args = (mu, sigma)); 
        recon_level = y1 / y2;
        
    recon_error, _ = quad(lambda x : (x - recon_level) ** 2 * norm.pdf(x, mu, sigma), a, b);
        
    return recon_level, recon_error;        
        
 
def MaxLloydQuantizer(func, limit_left, limit_right, N_levels, mu, sigma, tol = 10e-6):
    # Optimum iterative quantizer
    # Calculates and returns a tupil (quant_intervals, recon_level, recon_error)
    # func = sample pdf ('uniform', 'expon', 'gamma', 'rayleigh', 'norm')
    # limit_left = quantization left support region
    # limit_right = quantization right support region
    # N_levels = number of quantization levels
    # mu = mean
    # sigma = standard deviation
    # tol = tolerance
    
    step = (limit_right - limit_left) / N_levels;
    quant_intervals = [limit_left + step * i for i in range(N_levels)] + [limit_right];
    
    recon_levels = [ReconLevel(func, quant_intervals[i], quant_intervals[i + 1], mu, sigma)[0] for i in range(N_levels)];
    recon_error_old = limit_right - limit_left;
    recon_error = sum([ReconLevel(func, quant_intervals[i], quant_intervals[i + 1], mu, sigma)[1] for i in range(N_levels)]);
    
    while (abs(recon_error - recon_error_old) < tol * recon_error):
        recon_error_old = recon_error;
        quant_intervals = [limit_left] + [(recon_levels[i] + recon_levels[i + 1]) / 2 for i in range(N_levels - 1)] + [limit_right];
        recon_levels = [ReconLevel(func, quant_intervals[i], quant_intervals[i + 1], mu, sigma)[0] for i in range(N_levels)];
        recon_error = sum([ReconLevel(func, quant_intervals[i], quant_intervals[i + 1], mu, sigma)[1] for i in range(N_levels)]);
    
    return quant_intervals, recon_levels, recon_error

def ApplyQuantization(quant_intervals, recon_levels, signal_input):
    # Returns a tupil (signal_output, quant_error, SQNR)
    # quant_intervals = array of quantization intervals
    # recon_levels = array of quantization levels
    # signal_input = array of input signal
    
    signal_output = np.zeros(signal_input.shape);
    
    N_levels = len(recon_levels);
    for i in range(N_levels):
        if i != N_levels - 1:
            mask1 = signal_input >= quant_intervals[i];
            mask2 = signal_input < quant_intervals[i + 1];
            mask = mask1 * mask2;
        else:
            mask1 = signal_input >= quant_intervals[i];
            mask2 = signal_input <= quant_intervals[i + 1];
            mask = mask1 * mask2;
        
        signal_output[mask] = recon_levels[i];

    quant_error = signal_input - signal_output;
    SQNR = 20 * np.log10(LA.norm(signal_input) / LA.norm(quant_error));
    
    return signal_output, quant_error, SQNR;
    