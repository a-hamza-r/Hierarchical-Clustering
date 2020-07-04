# Python 3

import json
import gc
from datetime import datetime
from datetime import timedelta
import warnings
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")

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

def kmeans(epsilon =0):
        centroids = KMeans(n_clusters = 3)
        centroids.fit(df)
        return centroids
                
        
def plotnamed(data,centroids):
        color_dict = {'python':'r', 'gaussian':'y', 'mono': 'b'}
        x = data['wallclock_used']
        y = data['power']
        plt.scatter(x, y,color =[color_dict[i] for i in data['app_name']])
        plt.xlabel('Run_Time / Nodes')
        plt.ylabel('Power')
        plt.savefig('actualresults.png')

       
def plot(data,centroids):
        color_dict = ['r', 'y', 'b']
        x = data['wallclock_used']
        y = data['power']
        plt.scatter(x, y, color =[color_dict[i] for i in data['Assigned_Clus']])
        plt.xlabel('Run_Time / Nodes')
        plt.ylabel('Power')
        plt.savefig('2dim.png')

FMT = '%Y-%m-%dT%H:%M:%S.000Z'
LIMIT = 1000 # CHANGE THIS DEPENDING ON HOW MANY LINES YOU WANT TO READ FROM THE JSON FILE (MAX: 10,000)
c = 1
json_data = []
with open('10k.anon.json', 'rb') as f:
	for line in f:
		if(c <= LIMIT):
			json_data.append(json.loads(line.decode('utf-8').rstrip()))
			c += 1
		else:
			break
data = {'wallclock_used':[], 'power':[]}#, 'app_name':[], 'cpu_over_run':[]}
data2 = {'wallclock_used':[], 'power':[], 'app_name':[]}
for jd in json_data:
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
	if((jd['job']['app_name'] == "gaussian" or jd['job']['app_name'] == "mono" or jd['job']['app_name'] == "python") and (converttime(jd['job']['cpu_used']) != 0)):
		data['power'].append(np.average(power_list))#/(converttime(jd['job']['cpu_used'])))
		data2['power'].append(np.average(power_list))#/(converttime(jd['job']['cpu_used'])))
		t2 = jd['job']['end_time']
		t1 = jd['job']['start_time']
		data['wallclock_used'].append(converttime(jd['job']['wallclock_used'])/jd['job']['nodes_used'])
		data2['wallclock_used'].append(converttime(jd['job']['wallclock_used'])/jd['job']['nodes_used'])
		data2['app_name'].append(jd['job']['app_name'])
del json_data
gc.collect()
df = pd.DataFrame(data)
df2 = pd.DataFrame(data2)
del data
del data2
gc.collect()
df = df[df.power != 0]
df = df.dropna()
df2 = df2[df2.power != 0]
df2 = df2.dropna()
centroids = kmeans()
prediction = centroids.predict(df)
print("2-Dimension Clustering\n----------------------\nClusters:\n", centroids.cluster_centers_)
df2['Assigned_Clus']  = prediction
df3 = df2[df2.Assigned_Clus == 0]
df4 = df2[df2.Assigned_Clus == 1]
df5 = df2[df2.Assigned_Clus == 2]
pythoncount = len(df2[df2.app_name == 'python'].index)
monocount = len(df2[df2.app_name == 'mono'].index)
gauscount = len(df2[df2.app_name == 'gaussian'].index)
print("\nPython Count: ", pythoncount)
print("Mono Count: ", monocount)
print("Gaus Count: ", gauscount)

print("\nCLUSTER: 0  Count : ", len(df3.index))
print("PYTHON: ", len(df3[df3.app_name == 'python'].index)/len(df3.index))
print("MONO: ", len(df3[df3.app_name == 'mono'].index)/len(df3.index))
print("GAUS: ", len(df3[df3.app_name == 'gaussian'].index)/len(df3.index))

print("\nCLUSTER: 1  Count : ", len(df4.index))
print("PYTHON: ", len(df4[df4.app_name == 'python'].index)/len(df4.index))
print("MONO: ", len(df4[df4.app_name == 'mono'].index)/len(df4.index))
print("GAUS: ", len(df4[df4.app_name == 'gaussian'].index)/len(df4.index))

print("\nCLUSTER: 2  Count : ", len(df5.index))
print("PYTHON: ", len(df5[df5.app_name == 'python'].index)/len(df5.index))
print("MONO: ", len(df5[df5.app_name == 'mono'].index)/len(df5.index))
print("GAUS: ", len(df5[df5.app_name == 'gaussian'].index)/len(df5.index))
plot(df2, centroids)
plotnamed(df2,centroids)
