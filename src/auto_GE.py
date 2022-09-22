from lib2to3.pgen2 import grammar
from numpy.random import randint
from numpy.random import rand

import pygad
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
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
	config["fitness_function"] = int(config["fitness_function"])
	itr += 2

	# Generations ....................................................
	i = (individual[itr]%4) * 100
	if i==0 : i += ((individual[itr+1]%5)+5) * 10
	else : i += (individual[itr+1]%10) * 10
	i += (individual[itr+2]%10)
	config["num_generations"] = int(i) 			

	itr += 3
	
	# population .....................................................
	i = (individual[itr]%4) * 100
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
	solution, solution_fitness = image_segmentation.segmentation_GA.threshold_GA(config,histogram, num_thresholds)
	global best_solution_fitness, solution_thresholds, configuration
	if solution_fitness > best_solution_fitness:
		best_solution_fitness = solution_fitness
		solution_thresholds = solution
		configuration = config
	return solution_fitness

# automation ----------------------------------------------------------------------
def automated_GE(hist, num_thresh):
	global histogram, num_thresholds
	histogram = hist
	num_thresholds = num_thresh

	ga_instance = pygad.GA(num_generations=1,
                       num_parents_mating=2,
                       fitness_func=fitness_function,
                       sol_per_pop=50,
                       num_genes=18,
					   init_range_low=0,
                       init_range_high=100,
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

if __name__ == "__main__":
	print("-------------------------------------------------------------------------")
	print(test_individual)
	print(map_grammar(test_individual))
	print("-------------------------------------------------------------------------")