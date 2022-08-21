import pygad
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, io
from PIL import Image


#from image_segmentation.segmentation_GA import threshold_GA
import image_segmentation.segmentation_GA

# Genetic Algorithm  .............
def automated_GA(hist):

    #fitness_function = evaluation.evalution_function
    
    fitness_function = image_segmentation.segmentation_GA.threshold_GA
    image_segmentation.segmentation_GA.histogram = hist

    num_generations = 50
    num_parents_mating = 4

    sol_per_pop = 8
    num_genes = 11       # Number of thresholds
    gene_type = int

    gene_space =   [{'low': 10, 'high': 90, 'step': 1},  # Fitness
                    {'low': 50, 'high': 300, 'step': 1},    # Number of generation 
                    {'low': 50, 'high': 300, 'step': 1},    # Population size
                    {'low': 2, 'high': 3, 'step': 1},       # Number of parents mating
                    {'low': 0, 'high': 5, 'step': 1},       # Parent_selection_type
                    {'low': -1, 'high': 10, 'step': 1},     # keep_parents
                    {'low': 0, 'high': 3, 'step': 1},       # crossover_type
                    {'low': 10, 'high': 90, 'step': 1},     # crossover_probability
                    {'low': 0, 'high': 3, 'step': 1},       # mutation_type
                    {'low': 10, 'high': 90, 'step': 1},     # mutation_probability
                    {'low': 3, 'high': 10, 'step': 1},      # K_tournament
                ]

    parent_selection_type = "sss"
    keep_parents = 1

    crossover_type = "single_point"

    mutation_type = "random"
    mutation_probability = 0.2

    ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_function,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       gene_space=gene_space,
                       gene_type=gene_type,
                       parent_selection_type=parent_selection_type,
                       keep_parents=keep_parents,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_probability=mutation_probability)

    ga_instance.run()

    return ga_instance.best_solution()

def main():
    img = io.imread('./image_segmentation/images/baboon.png')
    img2 = np.array(Image.fromarray(img).convert('L'))
    hist, bins = np.histogram(img2, bins=range(256), density=False)

    f = automated_GA(hist)
    print(f)
    

if __name__ == "__main__":
    main()