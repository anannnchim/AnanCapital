#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-> Statistical Computing: Measuring Significant
"""

# library
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd 
np.random.seed(42)


# 1. Permutation test
def permutation_test(samples1, samples2, n_permutations):
    # Compute the observed difference of means
    observed_diff = np.mean(samples2) - np.mean(samples1)

    # Permutation test
    permuted_diffs = []
    combined_samples = np.concatenate((samples1, samples2))

    for _ in range(n_permutations):
        np.random.shuffle(combined_samples)
        new_diff = np.mean(combined_samples[len(samples1):]) - np.mean(combined_samples[:len(samples1)])
        permuted_diffs.append(new_diff)

    p_value = sum(np.abs(permuted_diffs) >= np.abs(observed_diff)) / n_permutations

    return observed_diff, p_value

'''
# Example usage

np.random.seed(42)
n_samples = 50
mean1, mean2 = 115.0, 120.0
std_dev = 10.0
n_permutations = 10000
samples1 = np.random.normal(mean1, std_dev, n_samples)
samples2 = np.random.normal(mean2, std_dev, n_samples)

 ----------------------------------------
 
observed_diff, p_value = permutation_test(samples1, samples2, n_permutations)
print(f"Observed difference of means: {observed_diff:.4f}")
print(f"P-value (permutation test): {p_value:.4f}")

# Plot the two distributions
x = np.linspace(mean1 - 3 * std_dev, mean2 + 3 * std_dev, 1000)
y1 = norm.pdf(x, mean1, std_dev)
y2 = norm.pdf(x, mean2, std_dev)

plt.plot(x, y1, color='blue', label=f'Mean {mean1}')
plt.plot(x, y2, color='orange', label=f'Mean {mean2}')
plt.legend()
plt.xlabel('Value')
plt.ylabel('Density')
plt.title('Normal Distributions')

plt.show()
'''




# 2. Permutation test

















