The aim of our project is to predict restaurant ratings in the Yelp dataset using multiclass Naive Bayes Classifier. We are predicting 'stars' for the businesses based on a selected feature set.

The repository is organized as below.
1. challenge - this folder contains our source .py files. The main file in the project is analysis.py which uses preprocessing.py and validation.py . extract_sentiment_value.py uses Stanford CORENLP to extract sentiment values for review. The results of teh snetiment analysis are saved into a data file and then used by preprocessing.py
