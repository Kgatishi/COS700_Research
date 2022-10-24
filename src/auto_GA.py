#from glob import glob
import pygad
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
from skimage import data, io
from PIL import Image
import time
from multiprocessing import Pool
from multiprocessing import Manager
from itertools import repeat

#from image_segmentation.segmentation_GA import threshold_GA
import sys
sys.path.append( './image_segmentation' )
import image_segmentation.segmentation_GA
# ............................................................

histogram = []
num_thresholds = 0
solution_thresholds = []
best_solution_fitness = -100
configuration = {}

# fitness function ...............
#def fitness_function(individual, individual_idx):
def fitness_function(individual, individual_idx):
    bitstring = [ int(i) for i in individual]
    config = {}
    # ----------------------------------------------------
    config["fitness_function"] = bitstring[0] * 0.01
    config["num_generations"] = bitstring[1]
    config["sol_per_pop"] = bitstring[2]              # population size max 100

    ps = ["sss"  , "rws" , "sus"  , "rank"  , "random"   , "tournament"  ]
    config["parent_selection_type"] = ps[ bitstring[3] ]
    config["K_tournament"] = bitstring[-1]

    ct = ["single_point", "two_points" , "uniform",  "scattered"] 
    config["crossover_type"] = ct[ bitstring[4] ]
    config["crossover_probability"] = bitstring[5] * 0.01

    mt = [ "random" , "inversion" , "scramble" , "swap" ]
    config["mutation_type"] = mt[ bitstring[6] ]
    config["mutation_probability"] =  bitstring[7] * 0.01

    # ----------------------------------------------------
    global num_thresholds, histogram
    #print(config)
    #print(num_thresholds)
    
    solution, solution_fitness = image_segmentation.segmentation_GA.threshold_GA(config, histogram, num_thresholds)
    '''
    global best_solution_fitness, solution_thresholds, configuration
    if solution_fitness > best_solution_fitness:
        best_solution_fitness = solution_fitness
        solution_thresholds = solution
        configuration = config
    '''
    #print(solution, solution_fitness)
    return config, solution, solution_fitness

# Genetic Algorithm  .............
#zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
def fitness_wrapper(solution):
    sol, globV = solution
    global histogram, num_thresholds
    histogram = globV['histogram']
    num_thresholds = globV['num_thresholds']
    #print(globV)
    return fitness_function(sol,0)


class PooledGA(pygad.GA):

    def cal_pop_fitness(self):
        global pool, histogram, num_thresholds
        #print("***", num_thresholds)
        manager = Manager()
        global_v = manager.dict({
            'histogram': histogram, 
            'num_thresholds': num_thresholds})
        pop_fitness_ = pool.map(fitness_wrapper, zip(self.population, repeat(global_v)))
        #print(pop_fitness_)
        
        pop_fitness = []
        for i in pop_fitness_:
            conf,sol, fit = i
            global best_solution_fitness, solution_thresholds, configuration
            if fit > best_solution_fitness:
                best_solution_fitness = fit
                solution_thresholds = sol
                configuration = conf
            pop_fitness.append(fit)

        pop_fitness = np.array(pop_fitness)
        return pop_fitness
#zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
def automated_GA(hist, num_thresh):

    global histogram, num_thresholds
    histogram = hist
    num_thresholds = num_thresh
    
    mutation_t = 3
    if num_thresholds <= 1: mutation_t = 2

    gene_space =   [{'low': 10, 'high': 91, 'step': 1},     # Fitness
                    {'low': 50, 'high': 300, 'step': 1},    # Number of generation 
                    {'low': 50, 'high': 300, 'step': 1},    # Population size
                    {'low': 0, 'high': 5, 'step': 1},       # Parent_selection_type
                    {'low': 0, 'high': 3, 'step': 1},       # crossover_type
                    {'low': 10, 'high': 90, 'step': 1},     # crossover_probability
                    {'low': 0, 'high': mutation_t, 'step': 1},       # mutation_type
                    {'low': 10, 'high': 90, 'step': 1},     # mutation_probability
                    {'low': 3, 'high': 10, 'step': 1},      # K_tournament
                ]

    ga_instance = PooledGA(num_generations=30,
                       num_parents_mating=2,
                       fitness_func=fitness_function,
                       sol_per_pop=30,
                       num_genes=9,
                       gene_space=gene_space,
                       gene_type=int,
                       parent_selection_type="sss",
                       keep_parents=2,
                       crossover_type="single_point",
                       mutation_type="random",
                       mutation_probability=0.2,
                       #parallel_processing=["thread", 3]
                       )
    #print( ga_instance.initial_population)
    #print(">>>>", num_thresholds)


    global pool
    with Pool(processes=4) as pool:
        ga_instance.run()

        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        global solution_thresholds, configuration
        return configuration, solution, solution_fitness, solution_thresholds
pool = 0
def main(im):

    img = io.imread(im)
    img2 = np.array(Image.fromarray(img).convert('L'))
    hist, bins = np.histogram(img2, bins=range(256), density=False)

    data = []
    list_thresholds = [1,3,5,7]
    for thres in list_thresholds:
        global solution_thresholds, configuration, best_solution_fitness
        solution_thresholds = []
        best_solution_fitness = -100
        configuration = {}

        t1 = time.time()
        config, solution, fitness, thresholds = automated_GA(hist,thres)
        
        t2 = time.time()
        
        print("-----------------------------------------------------------------------")
        print("Time is", t2-t1)
        print( thres, config, fitness, thresholds, t2-t1  )
        print("-----------------------------------------------------------------------")
        data.append( [thres, config, fitness, thresholds, t2-t1 ] )
        #break
    return data

if __name__ == "__main__":
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
        #break
    df = pd.DataFrame( results, columns=['num_threshold','config','fitness','thresholds','time'])
    df.to_csv("GA_results.csv")
        