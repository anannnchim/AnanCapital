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
import sys
from sklearn.utils import resample

# 1. Permutation test
def permutation_test(samples1, samples2, n_permutations):
    """
    Perform a permutation test to compare the means of two samples.

    Parameters:
        samples1 (array-like): The first sample.
        samples2 (array-like): The second sample.
        n_permutations (int): The number of permutations to perform.

    Returns:
        tuple: A tuple containing the observed difference of means and the p-value.

    """
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

# 2. Bootstrap: estimate uncertainty of mean or sd 
def bootstrap_standard_error(data, n_iterations=1000):
    means = []
    for _ in range(n_iterations):
        sample = resample(data)
        means.append(np.mean(sample))

    se = np.std(means, ddof=1)
    return se



# Example

'''
# Read the CSV file and extract the vector of data
csv_file = "/Users/nanthawat/Desktop/PythonExamPrep/Lecture/bootstrap_data.csv"
data_column = 'V1'
df = pd.read_csv(csv_file)
data = df[data_column].values

# Compute the bootstrap estimate of the standard error
se_estimate = bootstrap_standard_error(data)

print(f"Bootstrap estimate of the standard error: {se_estimate:.4f}")

# Calculate the mean value
mean_value = np.mean(data)

# Create a bar plot with the mean value and standard error
fig, ax = plt.subplots()
ax.bar(0, mean_value, yerr=se_estimate, capsize=10)
ax.set_xticks([0])
ax.set_xticklabels(['Mean Value'])
ax.set_title('Mean Value with Standard Error')

# Show the plot
plt.show()
'''

