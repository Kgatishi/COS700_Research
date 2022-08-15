import pygad
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, io
from PIL import Image

# Genetic Algorithm  .............
def automated_GA():

    #fitness_function = evaluation.evalution_function
    fitness_function = 0

    num_generations = 50
    num_parents_mating = 4

    sol_per_pop = 8
    num_genes = 4       # Number of thresholds

    init_range_low = -2
    init_range_high = 5

    parent_selection_type = "sss"
    keep_parents = 1

    crossover_type = "single_point"

    mutation_type = "random"
    mutation_percent_genes = 10

    ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_function,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       init_range_low=init_range_low,
                       init_range_high=init_range_high,
                       parent_selection_type=parent_selection_type,
                       keep_parents=keep_parents,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes)

    ga_instance.run()

    return ga_instance.best_solution()

def main():
    pass

if __name__ == "__main__":
    main()