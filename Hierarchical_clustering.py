import pandas as pd;
from os.path import join;
from numpy.linalg import norm;
import itertools;
import numpy as np;
import heapq;
from operator import add;
import math;
from collections import Counter;

'''
Apps that need to be excluded
Decide depending on the frequency of each app_name in the dataset:
amber: 45, fast: gamess: 12, gaussian: 47, ls-dyna: 10, matlab: 12, mfix: 18, mono: 192, python: 462, vasp: 985.
Rest of the apps have frequency less than 10. Those are excluded automatically
'''
appsToExclude = ['vasp', 'python', 'mono'];

'''
Number of samples to check. If set to a negative value, it will check for all the samples in the dataset, otherwise the set number of samples are checked
'''
samplesToConsider = -1;


class HierarchicalClustering:

	def __init__(self, dataFile, typeSimilarity):
		self.dataFile = dataFile;
		self.minHeap = [];
		self.typeSimilarity = typeSimilarity;

	def loadData(self):
		# dataDirectory = "Data";
		# df = pd.read_csv(join(dataDirectory, self.dataFile));
		df = pd.read_csv(self.dataFile);

		df = df[(df.mem_used != 0) & (df.power != 0) & (df.run_time != 0) & (df.nodes_used != 0) & (df.processors_used != 0) & (df.wallclock_used != 0) & (df.cpu_used != 0)]
		
		df1 = df.groupby('app_name').count().reset_index();
		appsToRemove = df1[df1.power < 10].app_name.unique().tolist();
		appsToRemove = appsToRemove + appsToExclude
		for x in appsToRemove:
			df = df[df.app_name != x];
		
		if samplesToConsider > -1:
			df = df.sample(n=samplesToConsider);
	
		self.numApps = len(df.app_name.unique());

		self.labels = df.pop("app_name").to_numpy();
		self.data = df;
		self.rows = df.to_numpy();

		self.proximities = [];
		self.clusterPairs = [];
		self.clusters = [];

	def computeProximities(self):
		for p1, p2 in itertools.combinations(range(len(self.rows)), 2):
			tupleClusters = (set([p1]), set([p2]));
			distancePts = norm(self.rows[p1]-self.rows[p2]); 
			self.proximities.append(distancePts);
			self.clusterPairs.append(tupleClusters);


	def proximity(self, clstr1, clstr2):
		if self.typeSimilarity == 0:
			return self.proximityGroupAvg(clstr1, clstr2);
		elif self.typeSimilarity == 1:
			return self.proximityMin(clstr1, clstr2);
		elif self.typeSimilarity == 2:
			return self.proximityMax(clstr1, clstr2);
		return -1;


	def proximityGroupAvg(self, clstr1, clstr2):
		sumProximity = 0;
		for p1, p2 in itertools.product(clstr1, clstr2):
			sumProximity += norm(self.rows[p1]-self.rows[p2]);
		proximityClusters = sumProximity/(len(clstr1)*len(clstr2));
		return proximityClusters;


	def proximityMax(self, clstr1, clstr2):
		maxDistance = 0;
		for p1, p2 in itertools.product(clstr1, clstr2):
			distPts = norm(self.rows[p1]-self.rows[p2]);
			if maxDistance < distPts:
				maxDistance = distPts;
		return maxDistance;


	def proximityMin(self, clstr1, clstr2):
		minDistance = math.inf;
		for p1, p2 in itertools.product(clstr1, clstr2):
			distPts = norm(self.rows[p1]-self.rows[p2]);
			if minDistance > distPts:
				minDistance = distPts;
		return minDistance;

	def findClusters(self):
		
		self.computeProximities();

		requiredClustersLeft = self.numApps*(self.numApps-1)/2;
		while len(self.clusterPairs) > requiredClustersLeft:
			minIdx = np.argmin(self.proximities);
			clstr1, clstr2 = self.clusterPairs[minIdx];
			del self.clusterPairs[minIdx];
			del self.proximities[minIdx];
			mergedClstr = clstr1.union(clstr2);
			length = len(self.clusterPairs);
			x = 0;
			#print(clstr1, clstr2);
			while x < length:
				if clstr1 in self.clusterPairs[x] or clstr2 in self.clusterPairs[x]:
					if clstr1 == self.clusterPairs[x][0] or clstr2 == self.clusterPairs[x][0]:
						newClstrPair1 = (mergedClstr, self.clusterPairs[x][1]);
						newClstrPair2 = (self.clusterPairs[x][1], mergedClstr);
					elif clstr1 == self.clusterPairs[x][1] or clstr2 == self.clusterPairs[x][1]:
						newClstrPair1 = (self.clusterPairs[x][0], mergedClstr);
						newClstrPair2 = (mergedClstr, self.clusterPairs[x][0]);
					if newClstrPair1 not in self.clusterPairs and newClstrPair2 not in self.clusterPairs:
						self.clusterPairs.append(newClstrPair1);
						self.proximities.append(self.proximity(*newClstrPair1));
					del self.clusterPairs[x];
					del self.proximities[x];
					length -= 1;
				else:
					x += 1;


def main():

	'''
	the constructor takes the filename as first argument and a number representing the proximity measure as the second argument. 
	For proximity measure, 0 corresponds to Group Average, 1 to Single Link (Min) and 2 to Complete Link (Max).	
	'''
	hClustering = HierarchicalClustering("10k.anon.csv", 0);
	hClustering.loadData();	
	
	hClustering.findClusters();

	clusters = [];
	for x in hClustering.clusterPairs:
		if x[0] not in clusters:
			clusters.append(x[0]);
		if x[1] not in clusters:
			clusters.append(x[1]);
	
	print("Hierarchical Clustering\n")

	i = 0;
	for x in clusters:
		actualApps = [];
		#center = [0, 0, 0, 0, 0, 0, 0];
		print("Cluster" + str(i) + ":", end=" ");
		i += 1;
		for y in x:
			#center = list(map(add, center, hClustering.rows[y]));
			#print(hClustering.labels[y], end=" ");
			actualApps.append(hClustering.labels[y]);
		#center = [z/len(x) for z in center];
		counter = Counter(actualApps);
		for c in counter:
			counter[c] = (counter[c], counter[c]/len(x));
			print(str(int(counter[c][1]*100)) + "% " + c + " (" + str(counter[c][0]) + "),", end=" ")
		
		print("")
		#print(center);
		#print("");

if __name__ == '__main__':
	main();
