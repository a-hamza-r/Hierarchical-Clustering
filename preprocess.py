# Python 3

# CIS5930 Data Mining project
# Remi Trettin, Ameer Hamza, Kenneth Burnham

# This script takes the '10k.anon.json' file as input (raw data),
# applies preprocessing techniques on the data, then outputs the
# new data as '10k.anon.csv' which significantly reduces the
# file size and improves data quality.

# Original dataset:
# 10k.anon.json.bz2 (631.72 MB)
# https://cscdata.nrel.gov/#/datasets/d332818f-ef57-4189-ba1d-beea291886eb

# IMPORTS

import json
import gc
from datetime import datetime

import pandas as pd
import numpy as np

# Given a dictionary, t, of time strings; convert it into seconds as
# an integer
def converttime(t):
	seconds = 0
	if(t is None):
		return seconds
	if("hours" in t):
		seconds += t['hours'] * 3600
	if("minutes" in t):
		seconds += t['minutes'] * 60
	if("seconds" in t):
		seconds += t['seconds']
	return seconds

FMT = '%Y-%m-%dT%H:%M:%S.000Z' # string time format
LIMIT = 10000 # change this value to limit how many data samples to read in. 10000 maximum
c = 1
json_data = []
with open('10k.anon.json', 'rb') as f:
	for line in f:
		if(c <= LIMIT):
			json_data.append(json.loads(line.decode('utf-8').rstrip())) # read in the json data, line by line until LIMIT is reached or EOF
			c += 1
		else:
			break

# Structured dictionary to hold data
data = {'power':[], 'run_time':[], 'nodes_used':[], 'processors_used':[], 'wallclock_used':[], 'cpu_used':[], 'mem_used':[], 'app_name':[]}

for jd in json_data: # for each data sample
	power_list = []
	for p in jd['power']:
		if(p is None):
			power_list.append(0)
		if("value" in p):
			power_list.append(p['value'])
		else:
			power_list.append(0)
	if(len(power_list) == 0):
		power_list = [0]
	data['power'].append(np.average(power_list)) # sample power usage = average of measured values or 0
	t2 = jd['job']['end_time']
	t1 = jd['job']['start_time']
	data['run_time'].append((datetime.strptime(t2, FMT) - datetime.strptime(t1, FMT)).total_seconds()) # given start_time and end_time, calculate job run time
	data['nodes_used'].append(jd['job']['nodes_used'])
	data['processors_used'].append(jd['job']['processors_used'])
	data['wallclock_used'].append(converttime(jd['job']['wallclock_used']))
	data['cpu_used'].append(converttime(jd['job']['cpu_used']))
	data['mem_used'].append(jd['job']['mem_used'])
	data['app_name'].append(jd['job']['app_name'])
del json_data
gc.collect()

df = pd.DataFrame(data) # organize the data dictionary into a dataframe
del data
gc.collect()
df = df[df.app_name != 'Unknown'] # remove samples with an unknown class/label/app_name
df = df.dropna() # remove samples with any empty value among its columns/features
df = df.sample(frac=1).reset_index(drop=True) # randomly shuffle the dataframe
print(df[:10].to_string()) # print first 10 samples
print("Number of classes:", len(df['app_name'].unique()))
print("Number of samples:", len(df.index))

df.to_csv('10k.anon.csv') # write the preprocessed data to a CSV for easier clustering/classification