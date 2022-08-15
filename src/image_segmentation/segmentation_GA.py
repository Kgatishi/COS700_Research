import pygad
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, io
from PIL import Image
#import evaluation

def GA(fitness):

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

def parameters(bitstring):
    return {bitstring[0]%1,
            bitstring[0]%1,
            bitstring[0]%1
            }


def main():
    coffee = data.coffee()
    img = io.imread('./images/baboon.png')
    
    
    print("coffee", coffee.shape)
    cof = np.array(Image.fromarray(coffee).convert('L'))
    print("reshape", cof.shape)
    print("reshape", cof.reshape(-1))

    hist, bins = np.histogram(cof, bins=range(256), density=True)
    print("Hists", hist.shape, hist.max(), hist.min())
    print("Bins", bins.shape, bins.max(), bins.min())


    print("++++++++++++++++++++++++++++++++++++")
    print("baboon", img.shape)
    img2 = np.array(Image.fromarray(img).convert('L'))
    print("baboon", img2.shape)
    print("reshape", img2.reshape(-1) )

    hist, bins = np.histogram(img2, bins=range(256), density=False)
    print("Hists", hist.shape, hist.max(), hist.min())
    print("Bins", bins.shape, bins.max(), bins.min())
    
    
    #print(hist)
    #f = Evaluation(hist)
    #GA(fitness=f)
    
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
    
if __name__ == "__main__":
    main()