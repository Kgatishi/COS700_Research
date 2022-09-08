from numpy.random import randint
from numpy.random import rand

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random 
import itertools 
import operator 
import numpy
import math

# 1. Selection operation: Tournament  ...........
def selection(pop, scores, k=3):
	tournament = randint(0, 20, k).tolist()
	
	hi, hi_s = 0, 0
	for i in range(1,k):
		if scores[tournament[hi]] < scores[tournament[i]]:
			hi_s = hi
			hi = i
	
	selrct = [pop[hi] , pop[hi_s] ]
	#print(selrct)
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
			bitstring[i] = randint(0, 20)

# 4. Genetic algorithm iterations ....
def genetic_algorithm(DATA, pop, objective, n_bits, n_iter, n_pop, r_cross, r_mut):
	
	best, best_eval = 0, objective(pop[0], DATA)
	scores = [objective(c,DATA) for c in pop]
	
	for gen in range(n_iter):
		
		
		std = np.std(scores)
		avg = np.mean(scores)
		max_ = np.max(scores)
		min_ = np.min(scores)
		
		print(">%d, avg = %.5f, std = %.20f, max = %.5f, min = %.5f" % (gen, avg, std, max_, min_  ))

		for i in range(n_pop):
			if scores[i] < best_eval:
				best, best_eval = pop[i], scores[i]

		select = selection(pop, scores) 
		#print(select)
		for c in crossover(select[0], select[1], r_cross):
			mutation(c, r_mut)
			score_ = objective(c, DATA)
			if min(scores) < score_ :
				indx = scores.index(min(scores))
				scores[indx] = score_
				pop[indx] = c

	return [pop, best, best_eval]
	
# 5. Evalution operation: ............
bnf_N = ["<expr>", "<op>", "<var>"]
bnf_T = ["0", "1", "2", "3", "4", "5", "+", "-", "/", "*", "x", "y", "z"]
bnf_S = ["<expr>"]
bnf_P ={"<expr>":[[ "<op>", "<expr>", "<expr>"],"<var>"],
		"<op>":["+", "-", "/", "*"],
		"<var>":["x", "y", "z", "0", "1", "2", "3", "4", "5"],
		}

bnf_expr = [[ "<op>", "<expr>", "<expr>"],"<var>"]
bnf_op = ["+", "-", "/", "*"]
bnf_var = ["x", "y", "z", "0", "1", "2", "3", "4", "5"]

def evalGrammaticTree( individual, data_samp):
		# Evaluate the sum of correctly identified loadshedding times
		result = sum( string_mapper(individual, dat[:1]) == dat[1] for dat in data_samp)
		return result / len(data_samp),

def string_mapper(individual, data):
	
	# Must be same shape/sie as bnf_var
	bnf_var_v = data + [ 1, 2, 3, 4, 5] 
	#print(bnf_var_v)
	size = len(individual)
	max_wrap = size *4

	itr = 0
	total_arr = []
	operator_arr = []
	expr_arr = [["<expr>",0]]
	while( len(expr_arr) > 0 and itr < max_wrap):
		#print(len(expr_arr))
		cur_expr = expr_arr.pop(0)
		depth = cur_expr[1]+1
		tree_choice = bnf_expr[ individual[itr%size] % len( bnf_expr ) ]
		itr += 1
		if isinstance(tree_choice, list):
			expr_arr.insert(0, ["<expr>",depth])
			expr_arr.insert(0, ["<expr>",depth])

			#insert operator: ["+", "-", "/", "*"] to operator_arr
			operator_arr.insert(0, bnf_op[individual[itr%size] % 4] )
			itr += 1
				
		else:
			num = bnf_var_v[ individual[itr%size] % len(bnf_var_v)]
			itr += 1
			if len(total_arr) > 0:
				loop = 0
				while loop< len(total_arr):
					var = total_arr[loop]
					
					if var[1] == depth :
						oper = operator_arr.pop(0)
						total_arr.pop(loop)
						if oper == "+":
							num += var[0]
						elif oper == "-":
							num -= var[0]
						elif oper == "/":
							if var[0] != 0 :
								num /= var[0]
						else:
							num *= var[0]
						depth -= 1
						loop = 0
					else: loop += 1
			total_arr.append([num,depth])

	if itr >= max_wrap:
		return -1

	tot = total_arr[0]
	return math.trunc((math.sin(tot[0])+1)*3)

# Read data from file and store in neccessary structures for 
def data_processing():
	
	df = pd.read_csv("archive/south_africa_load_shedding_history.csv")
	
	# Covert to datetime and sort by date
	df['DATE'] = pd.to_datetime(df.DATE)
	df.sort_values(by=["DATE"], inplace = True)
	heads = list(df.columns)
	df[heads] = df[heads].apply(pd.to_numeric, errors='coerce')
	return df

if __name__ == "__main__":
	print("----------------------------------------------------------------------------")
	print("0. Data Processing")
	print("----------------------------------------------------------------------------")
	df = data_processing()
	df_train = df.sample(frac=0.8,random_state=200) #random state is a seed value
	df_test = df.drop(df_train.index)

	print("Total Records:" + str(len(df)))
	print("Training Data:" + str(len(df_train)))
	print("Testing Data :" + str(len(df_test)))


	print("----------------------------------------------------------------------------")
	print("1. Training the GP for Decision tree")
	print("----------------------------------------------------------------------------")
	n_iter = 100                # total iterations
	n_bits = 20                 # bits string length
	n_pop = 100                 # population size
	r_cross = 0.8               # crossover rate
	r_mut = 1.0 / float(7) # mutation rate
	pop = [randint(0, 20, n_bits).tolist() for _ in range(n_pop)]

	# perform the genetic algorithm search
	df_train = df_train.values.tolist()
	DATA = list(df_train)
	#print(DATA)
	# results [pop, best, score ]
	results = genetic_algorithm(DATA, pop, evalGrammaticTree, n_bits, n_iter, n_pop, r_cross, r_mut)
	print('Done Training!')
	print("BEST >> " + str(results[1]) +" = " + str(results[2]) )


	print("----------------------------------------------------------------------------")
	print("2. Testing on unseen data")
	print("----------------------------------------------------------------------------")
	df_test = df_test.values.tolist()
	DATA = list(df_test)
	# results [pop, best, score ]
	results = genetic_algorithm(DATA, pop, evalGrammaticTree, n_bits, 1, n_pop, r_cross, r_mut)
	print('Done Testing!')
	print("BEST >> " + str(results[1]) +" = "+ str(results[2]) )


	print("----------------------------------------------------------------------------")
	print("THE END")
	print("----------------------------------------------------------------------------")