CIS5930 Project README
Remi Trettin, Ameer Hamza, Kenneth Burnham
---------------------------

Prerequisites:
	- Python 3

Install required libraries with:
	- pip install -r requirements.txt
	OR
	- python -m pip install -r requirements.txt

(Try python3 -m pip install -r requirements.txt if just python does not work)

10k.anon.json.bz2 file size: 631.72 MB
Dataset URL: https://cscdata.nrel.gov/#/datasets/d332818f-ef57-4189-ba1d-beea291886eb

The hpc_lightgbm.py script requires an external library (LightGBM).
The installation of this library varies based on platform.
See: https://github.com/microsoft/LightGBM
and: https://pypi.org/project/lightgbm/

---------------------------

preprocess.py
(The trimmed 10k.anon.csv file is provided therefore running this script is not required
to duplicate results.)
---------
Run with:
	- python preprocess.py

This script takes the raw data (10k.anon.json) as input, cleans it,
then writes a new file, 10k.anon.csv, which is much smaller for data
mining techniques.

---------------------------
10k.out.txt contains the result when the script is ran, showing each result for each program.


In order to run our code, you can individually run any of the commands within the bash script, if you only want to see one specific mining techinque, or if you want to see all of them print out sequentially and generate all of the graphs and scatterplots, just run the bash script.

./DMscript.bash

Again, each execution within the script will generate a graph for that mining technique, named appropriately. We kept them all in separate executions so that you have the ability to only run one that you specificly desire to see.

For each execution run, it takes roughly 27 seconds to pull all of the information from the .json file, and when the DMscript.bash is ran, it takes roughly 3 minutes for all executions to finish running and save results as images.

---------------------------
For hierarchical clustering (file: Hierarchical_clustering.py), we have defined two variables right after import statements, 'appsToExclude' and 'samplesToConsider'. It is possible to control which app_names to include in the algorithm. We have defined an array of app_names to be excluded from the algorithm, called appsToExclude. For now, it excludes vasp, python, mono. If you remove any app_name from there, it would be included in the algorithm, however the algorithm will be slower since these app_names are most frequent. The comment with the variable shows the frequency of each class in the processed dataset in order for you to have an idea about how long the code will take to run if you add any class to analysis. 
Another control is for number of samples. If the algorithm runs slow, you can set the number of samples to be considered in the algorithm and it will automatically run for required number of samples extracting random n samples from dataset without replacement if samplesToConsider=n. If this number is set to a negative value, the algorithm will consider all the samples.  
Secondly, it is possible to select which proximity measure to be used for the algorithm. In the main function, while creating an HierarchicalClustering object (line: 139), it is possible to pass a number representing proximity measure as the second argument. 0 corresponds to Group Average, 1 to Single Link (Min) and 2 to Complete Link (Max). 
You can check the implementations if required.