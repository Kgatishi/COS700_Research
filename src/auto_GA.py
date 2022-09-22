from glob import glob
import pygad
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, io
from PIL import Image

#from image_segmentation.segmentation_GA import threshold_GA
import image_segmentation.segmentation_GA
# ............................................................
histogram = 0
num_thresholds = 0
solution_thresholds = []
best_solution_fitness = -100
configuration = {}

# fitness function ...............
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
    solution, solution_fitness = image_segmentation.segmentation_GA.threshold_GA(config, histogram, num_thresholds)
    global best_solution_fitness, solution_thresholds, configuration
    if solution_fitness > best_solution_fitness:
        best_solution_fitness = solution_fitness
        solution_thresholds = solution
        configuration = config
        
    return solution_fitness

# Genetic Algorithm  .............
def automated_GA(hist, num_thresh):

    global histogram, num_thresholds
    histogram = hist
    num_thresholds = num_thresh
    
    mutation_t = 3
    if num_thresholds <= 1: mutation_t = 2

    gene_space =   [{'low': 10, 'high': 90, 'step': 1},     # Fitness
                    {'low': 50, 'high': 300, 'step': 1},    # Number of generation 
                    {'low': 50, 'high': 300, 'step': 1},    # Population size
                    {'low': 0, 'high': 5, 'step': 1},       # Parent_selection_type
                    {'low': 0, 'high': 3, 'step': 1},       # crossover_type
                    {'low': 10, 'high': 90, 'step': 1},     # crossover_probability
                    {'low': 0, 'high': mutation_t, 'step': 1},       # mutation_type
                    {'low': 10, 'high': 90, 'step': 1},     # mutation_probability
                    {'low': 3, 'high': 10, 'step': 1},      # K_tournament
                ]

    ga_instance = pygad.GA(num_generations=1,
                       num_parents_mating=2,
                       fitness_func=fitness_function,
                       sol_per_pop=50,
                       num_genes=9,
                       gene_space=gene_space,
                       gene_type=int,
                       parent_selection_type="sss",
                       keep_parents=2,
                       crossover_type="single_point",
                       mutation_type="random",
                       mutation_probability=0.2)
    ga_instance.run()

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    global solution_thresholds, configuration
    return configuration, solution, solution_fitness, solution_thresholds

def main():
    img = io.imread('./image_segmentation/images/baboon.png')
    img2 = np.array(Image.fromarray(img).convert('L'))
    hist, bins = np.histogram(img2, bins=range(256), density=False)

    f = automated_GA(hist)
    print(f)
    

if __name__ == "__main__":
    main()