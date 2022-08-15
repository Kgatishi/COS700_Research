from itertools import combinations
import numpy as np

"""Get the total entropy of regions for a given set of thresholds"""
def kapur_get_regions_entropy(hist, c_hist, thresholds):
    
    total_entropy = 0
    for i in range(len(thresholds) - 1):
        # Thresholds
        t1 = thresholds[i] + 1
        t2 = thresholds[i + 1]

        # print(thresholds, t1, t2)
        hc_val = c_hist[t2] - c_hist[t1 - 1]                        # Cumulative histogram
        h_val = hist[t1:t2 + 1] / hc_val if hc_val > 0 else 1       # Normalized histogram
        entropy = -(h_val * np.log(h_val + (h_val <= 0))).sum()     # entropy

        total_entropy += entropy

    return total_entropy
