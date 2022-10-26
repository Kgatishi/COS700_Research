from numpy.random import randint
from numpy.random import rand
from skimage import data, io
from PIL import Image

import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random

import sys
sys.path.append('./image_segmentation')
import image_segmentation.segmentation_GA


test_individual = [randint(0, 100) for i in range(25)]

histogram = 0
num_thresholds = 0
solution_thresholds = []
best_solution_fitness = -100
configuration = {}

mutation_t = 3


g_space =   [{'low': 10, 'high': 91, 'step': 1},     # Fitness
                    {'low': 50, 'high': 300, 'step': 1},    # Number of generation 
                    {'low': 50, 'high': 300, 'step': 1},    # Population size
                    {'low': 0, 'high': 5, 'step': 1},       # Parent_selection_type
                    {'low': 0, 'high': 3, 'step': 1},       # crossover_type
                    {'low': 10, 'high': 90, 'step': 1},     # crossover_probability
                    {'low': 0, 'high': mutation_t, 'step': 1},       # mutation_type
                    {'low': 10, 'high': 90, 'step': 1},     # mutation_probability
                    {'low': 3, 'high': 10, 'step': 1},      # K_tournament
                ]
# 0. fitness ----------------------------------------------------------------------
def fitness_function(individual, DATA):
    bitstring = [int(i) for i in individual]
    config = {}
    # ----------------------------------------------------
    config["fitness_function"] = bitstring[0] * 0.01
    config["num_generations"] = bitstring[1]
    config["sol_per_pop"] = bitstring[2]              # population size max 100

    ps = ["sss", "rws", "sus", "rank", "random", "tournament"]
    config["parent_selection_type"] = ps[bitstring[3]]
    config["K_tournament"] = bitstring[-1]

    ct = ["single_point", "two_points", "uniform",  "scattered"]
    config["crossover_type"] = ct[bitstring[4]]
    config["crossover_probability"] = bitstring[5] * 0.01

    mt = ["random", "inversion", "scramble", "swap"]
    config["mutation_type"] = mt[bitstring[6]]
    config["mutation_probability"] = bitstring[7] * 0.01

    # ----------------------------------------------------
    num_thresholds, histogram = DATA
    solution, solution_fitness = image_segmentation.segmentation_GA.threshold_GA(
        config, histogram, num_thresholds)

    global best_solution_fitness, solution_thresholds, configuration
    if solution_fitness > best_solution_fitness:
        best_solution_fitness = solution_fitness
        solution_thresholds = solution
        configuration = config

    #print(solution, solution_fitness)
    return solution_fitness

# 1. Selection operation: Tournament  ...........
def selection(pop, scores, k=6):
    tournament = randint(0, len(pop)-1, k).tolist()

    hi, hi_s = 0, 0
    for i in range(1, k):
        if scores[tournament[hi]] < scores[tournament[i]]:
            hi_s = hi
            hi = i

    selrct = [pop[hi], pop[hi_s]]
    return selrct


# 2. Crossover operation:  ...........
def crossover(p1, p2, r_cross):

    c1, c2 = p1.copy(), p2.copy()
    if rand() < r_cross:
        pt = randint(1, len(p1)-2)
        c1 = p1[:pt] + p2[pt:]
        c2 = p2[:pt] + p1[pt:]
    return [c1, c2]

# 3. Mutation operation:  ............
def mutation(bitstring, r_mut):
    for i in range(len(bitstring)):
        if rand() < r_mut:
            bitstring[i] = randint(g_space[i]["low"], g_space[i]["high"])

# 4. Genetic algorithm iterations ....
def genetic_algorithm(DATA, pop, objective, n_bits, n_iter, n_pop, r_cross, r_mut):
    global histogram, num_thresholds
    num_thresholds, histogram = DATA
    # print(DATA)
    best, best_eval = 0, objective(pop[0], DATA)
    scores = [objective(c, DATA) for c in pop]

    for gen in range(n_iter):

        std = np.std(scores)
        avg = np.mean(scores)
        max_ = np.max(scores)
        min_ = np.min(scores)

        #print(">%d, avg = %.5f, std = %.20f, max = %.5f, min = %.5f" % (gen, avg, std, max_, min_))

        for i in range(n_pop):
            if scores[i] > best_eval:
                best, best_eval = pop[i], scores[i]

        select = selection(pop, scores)
        
        for c in crossover(select[0], select[1], r_cross):
            
            mutation(c, r_mut)
            
            score_ = objective(c, DATA)
            if min(scores) < score_:
                indx = scores.index(min(scores))
                scores[indx] = score_
                pop[indx] = c

    global solution_thresholds, configuration
    return configuration, best, best_eval, solution_thresholds


def main(im):

    img = io.imread(im)
    img2 = np.array(Image.fromarray(img).convert('L'))
    hist, bins = np.histogram(img2, bins=range(256), density=False)

    # Parameters
    n_iter = 50                # total iterations
    n_bits = 9                 # bits string length
    n_pop = 50                 # population size
    r_cross = 0.7               # crossover rate
    r_mut = 1.0 / float(7)  # mutation rate
    
    

    # Get results for different number of threshold
    data = []
    list_thresholds = [1, 3, 5, 7]
    for thres in list_thresholds:
        global solution_thresholds, configuration, best_solution_fitness, g_space
        solution_thresholds = []
        best_solution_fitness = -100
        configuration = {}

        # New population for different number of thresholds
        global g_space
        if thres <= 1: g_space[6]["high"] = 2
        else: g_space[6]["high"] = 3
        pop = []
        for _ in range(n_pop):
            pop.append(
                [randint(g_space[prm]["low"], g_space[prm]["high"]) for prm in range(n_bits)] )
        
        
        DATA = thres, hist

        t1 = time.time()
        config, solution, fitness, thresholds = genetic_algorithm(
            DATA, pop, fitness_function, n_bits, n_iter, n_pop, r_cross, r_mut)
        t2 = time.time()
        print("-----------------------------------------------------------------------")
        print("Time is", t2-t1)
        print(thres, config, fitness, thresholds, t2-t1 )
        data.append([thres, config, fitness, thresholds, t2-t1])
        print("-----------------------------------------------------------------------")
        
    return data


if __name__ == "__main__":

    print("-------------------------------------------------------------------------")
    test_images = [
        './image_segmentation/images/baboon.png',
        './image_segmentation/images/Lenna.png',
        './image_segmentation/images/pepper.tiff',
        './image_segmentation/images/plane.png',
        './image_segmentation/images/house.tiff',
        './image_segmentation/images/pubbles.tiff',
    ]

    results = []
    for im in test_images:
        print("wmwmwmwmwmwmwmwmwmwmwmwmwmwmwmwmwmwmwm")
        print(im)
        im_results = main(im)
        for r in im_results:
            results.append(r)
        
    df = pd.DataFrame(results, columns=[
                      'num_threshold', 'config', 'fitness', 'thresholds', 'time'])
    df.to_csv("GE_results.csv")
