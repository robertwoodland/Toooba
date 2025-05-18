#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


resultsPerceptron = []

paths = os.listdir('./Perceptron/')
paths.sort(key=lambda p: tuple(map(int, p[:-4].split('_'))) if p.endswith('.txt') else (float('inf'),))
# paths.sort()
for name in paths:
    if os.path.isfile("./Perceptron/" + name):
        if name[-4:] == ".txt":
            res = pd.read_csv("./Perceptron/" + name, delimiter = "\\")
            result = ""
            for line in res.iloc[:, 0][1:]:
                try:
                    if line.startswith("instret:188799"):
                        result = line.split(" ")[16]
                        
                except:
                    pass
            if result != "":
                resultsPerceptron.append((name[:-4], result))
                    
resultsPerceptron


resultsTour = []

paths = os.listdir('./TourPred/')
paths.sort(key=lambda p: tuple(map(int, p[:-4].split('_'))) if p.endswith('.txt') else (float('inf'),))
# paths.sort()
for name in paths:
    if os.path.isfile("./TourPred/" + name):
        if name[-4:] == ".txt":
            res = pd.read_csv("./TourPred/" + name, delimiter = "\\")
            result = ""
            for line in res.iloc[:, 0][1:]:
                try:
                    if line.startswith("instret:188799"):
                        result = line.split(" ")[16]
                        
                except:
                    pass
            if result != "":
                resultsTour.append((name[:-4], result))
                    
resultsTour


resultsBht = []

paths = os.listdir('./Bht/')
paths.sort(key=lambda p: tuple(map(int, p[:-4].split('_'))) if p.endswith('.txt') else (float('inf'),))
# paths.sort()
for name in paths:
    if os.path.isfile("./Bht/" + name):
        if name[-4:] == ".txt":
            res = pd.read_csv("./Bht/" + name, delimiter = "\\")
            result = ""
            for line in res.iloc[:, 0][1:]:
                try:
                    if line.startswith("instret:188799"):
                        result = line.split(" ")[16]
                        
                except:
                    pass
            if result != "":
                resultsBht.append((name[:-4], result))
                    
resultsBht


# Plot graph!
plt.figure(figsize=(10, 6))
plt.title("CoreMark CPU Cycle Count for Different Branch Predictors")
plt.xlabel("Hardware Budget (Bytes)")
plt.ylabel("Cycle Count")

# Tournament
x = []
y = []
for result in resultsTour:
    x.append(result[0].split("_")[0])
    y.append(float(result[1]))
plt.plot(range(len(x)), y, marker='o', markersize=4, linestyle='-', linewidth=1, color='g', label='Tournament')

# BHT
x = []
y = []
for result in resultsBht:
    x.append(result[0].split("_")[0])
    y.append(float(result[1]))
plt.plot(range(len(x)), y, marker='o', markersize=4, linestyle='-', linewidth=1, color='r', label='BHT')

# Perceptron
x = []
y = []
for result in resultsPerceptron:
    x.append(result[0].split("_")[0])
    y.append(float(result[1]))
plt.plot(range(len(x)), y, marker='o', markersize=4, linestyle='-', linewidth=1, color='b', label='Perceptron')

plt.xticks(range(len(x)), x)

# Add a legend
plt.legend(loc='upper right')

# plt.xticks(x, rotation=45)
plt.grid()
plt.tight_layout()
plt.savefig("branch_prediction_accuracy.png", dpi=500)
plt.show()




