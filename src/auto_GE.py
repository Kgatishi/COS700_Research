from lib2to3.pgen2 import grammar
from numpy.random import randint
from numpy.random import rand
from skimage import data, io
from PIL import Image

import pygad
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import time
import time
from multiprocessing import Pool
from multiprocessing import Manager
from itertools import repeat

import sys
sys.path.append( './image_segmentation' )
import image_segmentation.segmentation_GA

test_individual = [ randint(0,100) for i in range(25)]

terminal_set = []
functional_set = []

grammar_p = {
		"<parameter>":["<population><selection><crossover><probability><mutation><probability>"],
		"<selection>":["<sss>", "<rws>", "<sus>", "<rank>","<random>", "<K_tournament>"],
		"<crossover>":["single_point", "two_points" , "uniform",  "scattered"],
		"<mutation>":["random" , "swap" , "inversion" , "scramble"],
		"<population>": ["<v><v>","1<v><v>","2<v><v>","3<v><v>","4<v><v>"],
		"<probability>": ["0.<v><v>"],
		"<K_tournament>":["<v>","<1><v>"],
		"<v>":[ "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
}

histogram = 0
num_thresholds = 0
solution_thresholds = []
best_solution_fitness = -100
configuration = {}

# grammar ....................................................................................
def map_grammar(individual, data=0):
	itr = 0
	config = {}
	# Fitness ....................................................
	config["fitness_function"] = (individual[itr]%10)*0.1
	config["fitness_function"] += (individual[itr+1]%10)*0.01
	itr += 2

	# Generations ....................................................
	i = (individual[itr]%4) * 100
	if i==0 : i += ((individual[itr+1]%5)+5) * 10
	else : i += (individual[itr+1]%10) * 10
	i += (individual[itr+2]%10)
	config["num_generations"] = int(i) 			

	itr += 3
	
	# population .....................................................
	i = (individual[itr]%3) * 100
	if i==0 : i += ((individual[itr+1]%5)+5) * 10
	else : i += (individual[itr+1]%10) * 10
	i += (individual[itr+2]%10)
	config["sol_per_pop"] = int(i)
			
	itr += 3

	# selection ......................................................
	selection = ["sss", "rws", "sus", "rank","random", "tournament"]
	config["parent_selection_type"] = selection[individual[itr]%6]
	
	if config["parent_selection_type"] == "tournament": 
		config["K_tournament"] = int((individual[itr+1]%6)+3)
		itr += 2
	else: 
		config["K_tournament"] = 0
		itr += 1
	# crossover ......................................................
	crossover = ["single_point", "two_points" , "uniform",  "scattered"]
	config["crossover_type"] = crossover[ individual[itr]%4 ]
	config["crossover_probability"] = (individual[itr+1]%10)*0.1
	config["crossover_probability"] += (individual[itr+2]%10)*0.01
	itr += 3
	# mutation ........................................................
	mutation = ["random" , "inversion" , "scramble", "swap" ]
	global num_thresholds
	if num_thresholds <=1 : config["mutation_type"] = mutation[individual[itr]%3]
	else : config["mutation_type"] = mutation[individual[itr]%4]
	
	config["mutation_probability"] = (individual[itr+1]%10)*0.1
	config["mutation_probability"] += (individual[itr+2]%10)*0.01
	itr += 3

	return config

# fitness ----------------------------------------------------------------------
def fitness_function(individual, individual_idx):

	config = map_grammar(individual=individual)
	global num_thresholds, histogram

	solution, solution_fitness = image_segmentation.segmentation_GA.threshold_GA(config,histogram, num_thresholds)
	'''
	global best_solution_fitness, solution_thresholds, configuration
	if solution_fitness > best_solution_fitness:
		best_solution_fitness = solution_fitness
		solution_thresholds = solution
		configuration = config
	'''
	return config, solution, solution_fitness
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
# automation ----------------------------------------------------------------------
def automated_GE(hist, num_thresh):
	global histogram, num_thresholds
	histogram = hist
	num_thresholds = num_thresh

	ga_instance = PooledGA(num_generations=1,
                       num_parents_mating=2,
                       fitness_func=fitness_function,
                       sol_per_pop=5,
                       num_genes=18,
					   init_range_low=0,
                       init_range_high=100,
                       gene_type=int,
                       parent_selection_type="sss",
                       keep_parents=2,
                       crossover_type="single_point",
                       mutation_type="random",
                       mutation_probability=0.2)

	global pool
	with Pool(processes=10) as pool:
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
    list_thresholds = [5,1,3,5,7]
    for thres in list_thresholds:
        global solution_thresholds, configuration, best_solution_fitness
        solution_thresholds = []
        best_solution_fitness = -100
        configuration = {}

        t1 = time.time()
        config, solution, fitness, thresholds = automated_GE(hist,thres)
        t2 = time.time()
        print("-----------------------------------------------------------------------")
        print("Time is", t2-t1)
        data.append( [thres, config, fitness, thresholds, t2-t1 ] )
        break
    return data

if __name__ == "__main__":
	print("-------------------------------------------------------------------------")
	print(test_individual)
	print(map_grammar(test_individual))
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
	df = pd.DataFrame( results, columns=['num_threshold','config','fitness','thresholds','time'])
	df.to_csv("GE_results.csv")