The aim of our project is to predict restaurant ratings in the Yelp dataset using multiclass Naive Bayes Classifier. We are predicting 'stars' for the businesses based on a selected feature set.

The repository is organized as below.

1. challenge - This folder contains our source .py files. The main file in the project is analysis.py which uses preprocessing.py and validation.py . extract_sentiment_value.py uses Stanford CORENLP to extract sentiment values for review. The results of the snetiment analysis are saved into a data file and then used by preprocessing.py
2. demo -  This folder contains our ipython notebooks which we used extensively for our experiments. The file Analysis.ipynb shows our latest merged experiment results. The file analysis_sklearn.ipynb shows the sklearn comparison. The file visualization.ipynb contains all the plots included in the final report
3. out - This folder contains all our intermediate preprocessed data files. All of them are necessary for the code to run.
4. preprocessing - This folder contains our very initial experiments to understand the data. Not very relevant to the classifier.
5. report - This folder contains all the reports for this project.
6. tests - This folder contains all the unittests. analysis_test.py methods will internally test both analysis.py and preprocessing.py core methods. validation_test.py tests validation.py .
7. contributions.txt has the details of contributions from each team member to the project.

How to run the code and tests ?

1. Download yelp_academic_dataset_business.json and put this in the root folder
2. Open bash to the root of the repo and issue this command to set up the path - 

export PYTHONPATH="`pwd`/challenge:`pwd`/tests"

Following command will run the tests

python -m unittest validation_test

python -m unittest analysis_test

Following command will run the main file and print the classifier accuracy.

python challenge/analysis.py

