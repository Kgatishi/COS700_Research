from glob import glob
import pygad
import random
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, io
from PIL import Image
#from evaluation import *
from . import evaluation

histogram = 0

def threshold_GA(bitstring, solution_idx=0):
    bitstring = [ int(i) for i in bitstring]
    print(bitstring)
    print(type(bitstring[0]))
    # No automation 
    # ----------------------------------------------------
    #fitness_function = evaluation.evalution_function
    fitness_function = evaluation.evalution_function
    num_genes = 4       # Number of thresholds
    gene_type = int     # Might not be neccesasry if 
    
    init_range_low = 0
    init_range_high = 255

    # No automation 
    # ----------------------------------------------------
    global histogram
    evaluation.image_histograms = histogram
    global OTSU
    OTSU = bitstring[0] * 0.01
    global KAPUR
    KAPUR = 1- OTSU

    num_generations = bitstring[1]
    sol_per_pop = bitstring[2]           # population size max 100
    num_parents_mating = bitstring[3] 

    #steady-state, roulette wheel, stochastic universal, rank, random, tournament
    ps = ["sss"  , "rws" , "sus"  , "rank"  , "random"   , "tournament"  ]
    parent_selection_type = ps[ bitstring[4] % len(ps) ]
    keep_parents = bitstring[5]
    K_tournament = bitstring[-1]


    ct = ["single_point", "two_points" , "uniform",  "scattered"] 
    crossover_type = ct[ bitstring[6] % len(ct)]
    crossover_probability = bitstring[7] * 0.01

    mt = [ "random" , "swap" , "inversion" , "scramble" ]
    mutation_type = mt[ bitstring[8] % len(mt)]
    mutation_probability =  bitstring[9] * 0.01


    ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_function,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       init_range_low=init_range_low,
                       init_range_high=init_range_high,
                       gene_type=gene_type,
                       allow_duplicate_genes=False,
                       parent_selection_type=parent_selection_type,
                       keep_parents=keep_parents,
                       crossover_type=crossover_type,
                       crossover_probability=crossover_probability,
                       mutation_type=mutation_type,
                       mutation_probability=mutation_probability)

    ga_instance.run()
    print(ga_instance.best_solution())

    chr, fit, idx = ga_instance.best_solution()
    return fit
    #return ga_instance.best_solution()


def main():
    img = io.imread('./images/baboon.png')


    print("++++++++++++++++++++++++++++++++++++")
    print("baboon", img.shape)
    img2 = np.array(Image.fromarray(img).convert('L'))
    print("baboon", img2.shape)
    print("reshape", img2.reshape(-1) )

    hist, bins = np.histogram(img2, bins=range(256), density=False)
    print("Hists", hist.shape, hist.max(), hist.min())
    print("Bins", bins.shape, bins.max(), bins.min())
    
    num_p_mate = random.randint(2, 3+1)
    gene_space =   [random.randint(10, 90+1) ,  # Fitness
                    random.randint(50, 300+1) ,    # Number of generation 
                    random.randint(50, 300+1) ,    # Population size
                    num_p_mate ,       # Number of parents mating
                    random.randint(0, 5+1) ,       # Parent_selection_type
                    num_p_mate ,     # keep_parents
                    random.randint(0, 3+1) ,       # crossover_type
                    random.randint(10, 90+1) ,     # crossover_probability
                    random.randint(0, 3+1) ,       # mutation_type
                    random.randint(10, 90+1) ,     # mutation_probability
                    random.randint(3, 10+1)      # K_tournament
    ]
    #gene_space = [,,,,,,,,]
    #print(hist)
    print(gene_space)
    global histogram
    histogram = hist
    f = threshold_GA(bitstring=gene_space)
    print(f)
    '''
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 4))
    ax = axes.ravel()

    print("---------------------------------")
    ax[0].imshow(img, cmap=plt.cm.gray, vmin=0, vmax=1)
    ax[0].set_title('Original Image')
    print("-----------")
    ax[1].imshow(img2, cmap=plt.cm.gray, vmin=0, vmax=255)
    ax[1].set_title('Gray Image')
    print("-----------")
    ax[2].bar(bins[:-1],height=hist)
    ax[2].set_title('Histogram')
    print("-----------")
    plt.tight_layout()
    plt.show()
    '''
if __name__ == "__main__":
    main()