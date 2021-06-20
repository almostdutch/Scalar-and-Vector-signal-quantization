'''
vector_quantization_utils.py

'''

import numpy as np
from scipy.stats import uniform, expon, gamma, rayleigh, norm     
import random
import math
from numpy import linalg as LA

def KmeansQuantizer(func, N_iter, N_samples, N_dim, N_levels, mu, sigma, tol = 10e-6):
    # Optimum iterative quantizer
    # Calculates and returns a tupil (recon_levels, recon_error)
    # func = pdf ('uniform', 'expon', 'gamma', 'rayleigh', 'norm')
    # N_iter = number of random initializations for N_levels recon levels
    # N_samples = number of training samples drawn from the func pdf
    # N_dim = dimentionality of quantizer
    # N_levels = number of quantization levels
    # mu = pdf mean
    # sigma = pdf standard deviation
    # tol = tolerance
    
    recon_levels_N_iter = np.zeros((N_iter, N_levels, N_dim), dtype = np.float32, order = 'C');
    recon_error_N_iter = np.zeros((N_iter, 1), dtype = np.float32, order = 'C');
    
    for iter_no in range(N_iter):
        random.seed(a = None, version = 2);
        N_groups = math.floor(N_samples / N_dim);
        training_samples = eval(f'{func}.rvs(size = N_samples, loc = mu, scale = sigma)'); 
        training_samples = training_samples.reshape(N_groups, N_dim, order = 'C');
        training_samples_ownership = np.zeros((1, N_groups), order = 'C');
        initial_recon_levels_no = [random.randint(0, N_groups - 1) for i in range(N_levels)];
        recon_levels = training_samples[initial_recon_levels_no]; 
        recon_error_old = N_samples;
        recon_error = 0;
        
        # assign samples to their clusters
        for i in range(N_groups):
            sample = training_samples[i];
            error = np.sum((sample - recon_levels) ** 2, axis = 1);
            ind_min = np.argmin(error, axis = 0);
            training_samples_ownership[0, i] = ind_min
            recon_error += error[ind_min];
        
        # update clusters
        for i in range(N_levels):
            mask = training_samples_ownership == i;
            N_elements = np.sum(mask);
            if N_elements > 0:
                recon_levels[i] = mask @ training_samples / N_elements;
        
        while (abs(recon_error - recon_error_old) < tol * recon_error):
            
            recon_error_old = recon_error;
            recon_error = 0;
            
            # assign samples to their clusters
            for i in range(N_groups):
                sample = training_samples[i];
                error = np.sum((sample - recon_levels) ** 2, axis = 1);
                ind_min = np.argmin(error, axis = 0);
                training_samples_ownership[0, i] = ind_min
                recon_error += error[ind_min];
            
            # update clusters
            for i in range(N_levels):
                mask = training_samples_ownership == i;
                N_elements = np.sum(mask);
                if N_elements > 0:
                    recon_levels[i] = mask @ training_samples / N_elements;

        # book keeping
        recon_levels_N_iter[iter_no] = recon_levels;
        recon_error_N_iter[iter_no] = recon_error;  
        
    # choose the recon levels with the lowerst recon error
    ind_min = np.argmin(recon_error_N_iter, axis = 0);
    recon_levels = recon_levels_N_iter[ind_min].reshape(N_levels, N_dim);
    recon_error = recon_error_N_iter[ind_min].item();
    
    return recon_levels, recon_error

def ApplyQuantization(recon_levels, signal_input):
    # Returns a tupil (signal_output, quant_error, SQNR)
    # recon_levels = array of quantization levels
    # signal_input = array of input signal
    
    signal_output = np.zeros(signal_input.shape);
    N_dim = recon_levels.shape[1];
    N_samples = len(signal_input);
    N_groups = int(N_samples / N_dim);
    
    for i in range(N_groups):
        # assign samples to their clusters
        sample = signal_input[i * N_dim: i * N_dim + N_dim];
        error = np.sum((sample - recon_levels) ** 2, axis = 1);
        ind_min = np.argmin(error, axis = 0);
        signal_output[i * N_dim: i * N_dim + N_dim] = recon_levels[ind_min];
        
    quant_error = signal_input - signal_output;
    SQNR = 20 * np.log10(LA.norm(signal_input) / LA.norm(quant_error));
    
    return signal_output, quant_error, SQNR;
