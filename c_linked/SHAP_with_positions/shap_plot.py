import shap
import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

all_shap_values = np.loadtxt('shap_values.csv', delimiter=',')
sample_count = all_shap_values.shape[0]

# sum SHAP values for each feature
shap_values = np.sum(all_shap_values, axis=0)

print(shap_values.shape)

shap_values = shap_values/sample_count

closest_one = shap_values[6] + shap_values[8]
closest_two = shap_values[5] + shap_values[9]
closest_three = shap_values[4] + shap_values[10]
closest_four = shap_values[3] + shap_values[11]
closest_five = shap_values[2] + shap_values[12]
closest_six = shap_values[1] + shap_values[13]
closest_seven = shap_values[0] + shap_values[14]

neighbours = [closest_one, closest_two, closest_three, closest_four, closest_five, closest_six, closest_seven]

# Create the figure
plt.figure(figsize=(10, 6))

# Bar plot with enhanced appearance
bars = plt.bar(range(7), neighbours, color='skyblue', edgecolor='black', linewidth=1.2)

# Customize the ticks
plt.xticks(range(7), ['Closest 1 (6&8)', 'Closest 2 (5&9)', 'Closest 3 (4&10)', 'Closest 4 (3&11)', 'Closest 5 (2&12)', 'Closest 6 (1&13)', 'Closest 7 (0&14)'], rotation=45, ha='right')

# Add grid for readability
plt.grid(True, axis='y', linestyle='--', alpha=0.7)

# Add labels and title
plt.xlabel('Neighbouring residues', fontsize=12, fontweight='bold')
plt.ylabel('Absolute Averaged SHAP Values', fontsize=12, fontweight='bold')
plt.title('Absolute Averaged SHAP Values vs Neighbouring residues', fontsize=14, fontweight='bold')

# Adjust the layout for better spacing
plt.tight_layout()

# Show the plot
plt.show()
