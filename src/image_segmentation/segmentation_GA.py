from glob import glob
import pygad
import random
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, io
from PIL import Image
#from evaluation import *
import evaluation



def threshold_GA(config, img_hist, num_thresholds ):
    
    # ----------------------------------------------------
    fitness_function = evaluation.evalution_function
    num_genes = num_thresholds   
    evaluation.image_histograms = img_hist
    evaluation.OTSU = config["fitness_function"]
    evaluation.KAPUR = 1- config["fitness_function"]
    #print(" --- num_thresholds ---", num_thresholds)
    # ----------------------------------------------------

    ga_instance = pygad.GA(num_generations = config["num_generations"],
                       num_parents_mating=2,
                       fitness_func = fitness_function,
                       sol_per_pop = config["sol_per_pop"],
                       num_genes=num_genes,
                       init_range_low=0,
                       init_range_high=255,
                       gene_type=int,
                       allow_duplicate_genes = False,
                       parent_selection_type = config["parent_selection_type"],
                       K_tournament = config["K_tournament"],
                       crossover_type = config["crossover_type"],
                       crossover_probability = config["crossover_probability"],
                       mutation_type = config["mutation_type"],
                       mutation_probability = config["mutation_probability"])

    ga_instance.run()
    
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    return solution, solution_fitness


def main():
    img = io.imread('./images/baboon.png')
    img2 = np.array(Image.fromarray(img).convert('L'))
    hist, bins = np.histogram(img2, bins=range(256), density=False)

    print("Hists", hist.shape, hist.max(), hist.min())
    print("Bins", bins.shape, bins.max(), bins.min())
    
    gene_space = {
                    "fitness_function": 0,          # random.randint(10, 90+1) ,
                    "num_generations": 50 ,         # random.randint(50, 300+1) ,
                    "sol_per_pop": 50,              # random.randint(50, 300+1) ,
                    "parent_selection_type": "sss", # ["sss","rws","sus","rank","random" ,"tournament"]
                    "crossover_type": "single_point", # ["single_point","two_points","uniform","scattered"] 
                    "crossover_probability": 0.1 ,  # [0-1]
                    "mutation_type": "random",             # ["random","inversion","scramble","swap"]
                    "mutation_probability": 0.1 ,   # [0-1]
                    "K_tournament": 3               # [3-10]
    }
    
    #print(hist)
    print(gene_space)
    f = threshold_GA(config=gene_space, img_hist=hist, num_thresholds=4)
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