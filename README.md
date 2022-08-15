# COS700_Research
Automated Design of GA for Image Segmentation
## Matrices
* Evaluation
	* Otsu
	* Kapur
* Objective
	* PSMR (Peak signal-to-noise ratio)
	* SSIM (Structural Similarity Index)

## Image Segmentation
### Genetic Agorithm

| Representation    | Description |
| ------------------|:-------:|
| Chromosome        |   No. of thresholds    |
| Gene              |   Threshold value   |


* Parameters
	* Population size
	* Number of generations
	* Selection 
	     * Method: Tournament, Roulette Wheel, Elitism
	* Crossover
	     * Rate: Percentage
	     * Method: Single-point, k-point, Uniform,
	* Mutation
	     * Rate: Percentage
	     * Method: Random (no. points)
	* Evaluation: Ostu, Kapur

## Automation
 - Multipoint Search - Gentic Algorithm
 - Single Point Search - Simulated Annealing

## Results
