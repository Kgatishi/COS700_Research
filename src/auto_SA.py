import pygad
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, io
from PIL import Image
from scipy import optimize
from itertools import combinations

# Simulated Annealing .............
def automated_SA():
    
    np.random.seed(555)   # Seeded to allow replication.
    x0 = np.array([2., 2.])     # Initial guess.
    nthrs = 1
    thr_combinations = combinations(range(255), nthrs)
    x0 = thr_combinations[6]     # Initial guess.
    res = optimize.anneal(f, x0, args=histogram, schedule='boltzmann',
                          full_output=True, maxiter=500, lower=-10,
                          upper=10, dwell=250, disp=True)
    res[0]  # obtained minimum
    res[1]  # function value at minimum
    return res[0] , res[1]


    
def main():
    pass

if __name__ == "__main__":
    main()