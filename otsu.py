from itertools import combinations
import numpy as np


def otsu_get_variance(hist, c_hist, cdf, thresholds):
    """Get the total entropy of regions for a given set of thresholds"""

    variance = 0

    for i in range(len(thresholds) - 1):
        # Thresholds
        t1 = thresholds[i] + 1
        t2 = thresholds[i + 1]

        weight = c_hist[t2] - c_hist[t1 - 1]                # Cumulative histogram
        r_cdf = cdf[t2] - cdf[t1 - 1]                       # Region CDF
        r_mean = r_cdf / weight if weight != 0 else 0       # Region mean

        variance += weight * r_mean ** 2

    return variance
