#!/bin/bash

# K-Means
python3 7dim.py
echo ----------------------------------
python3 5dim.py
echo ----------------------------------
python3 2dim.py
echo ----------------------------------
python3 normal2dim.py
echo ----------------------------------

# Classification
python3 hpc_classify.py
echo ----------------------------------
python3 hpc_lightgbm.py
echo ----------------------------------

# Hierarchical Clustering
python3 Hierarchical_clustering.py
echo ----------------------------------
