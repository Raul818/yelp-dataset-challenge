# Naive Bayes Classifier for Yelp Business Rating

The aim of our project is to predict restaurant ratings in the Yelp dataset using multiclass Naive Bayes Classifier. We are predicting 'stars' for the businesses based on a selected feature set.

The repository is organized as below.

- **challenge** - This folder contains our source .py files. The main file in the project is [`analysis.py`](https://github.com/ljishen/yelp-dataset-challenge/blob/master/challenge/analysis.py) which uses [`preprocessing.py`](https://github.com/ljishen/yelp-dataset-challenge/blob/master/challenge/preprocessing.py) and [`validation.py`](https://github.com/ljishen/yelp-dataset-challenge/blob/master/challenge/validation.py). [`extract_sentiment_value.py`](https://github.com/ljishen/yelp-dataset-challenge/blob/master/challenge/extract_sentiment_value.py) uses StanfordCoreNLP  to extract sentiment values for reviews and tips. The results of the snetiment analysis are saved into the `out/yelp_academic_dataset_review_sentiment.json` and `out/yelp_academic_dataset_tip_sentiment.json` and then used by [`preprocessing.py`](https://github.com/ljishen/yelp-dataset-challenge/blob/master/challenge/preprocessing.py)
- **demo** - This folder contains our ipython notebooks which we used extensively for our experiments. The file [`Analysis.ipynb`](https://github.com/ljishen/yelp-dataset-challenge/blob/master/demo/Analysis.ipynb) shows our latest merged experiment results. The file [`analysis_sklearn.ipynb`](https://github.com/ljishen/yelp-dataset-challenge/blob/master/demo/analysis_sklearn.ipynb) shows the sklearn comparison. The file [`visualization.ipynb`](https://github.com/ljishen/yelp-dataset-challenge/blob/master/demo/visualization.ipynb) contains all the plots included in the final report
- **out** - This folder contains all our intermediate preprocessed data files. All of them are necessary for the code to run.
- **preprocessing** - This folder contains our very initial experiments to understand the data. Not very relevant to the classifier.
- **report** - This folder contains all the reports for this project.
- **tests** - This folder contains all the unittests. [`analysis_test.py`](https://github.com/ljishen/yelp-dataset-challenge/blob/master/tests/analysis_test.py) methods will internally test both [`analysis.py`](https://github.com/ljishen/yelp-dataset-challenge/blob/master/challenge/analysis.py) and [`preprocessing.py`](https://github.com/ljishen/yelp-dataset-challenge/blob/master/challenge/preprocessing.py) core methods. [`validation_test.py`](https://github.com/ljishen/yelp-dataset-challenge/blob/master/tests/validation_test.py) tests [`validation.py`](https://github.com/ljishen/yelp-dataset-challenge/blob/master/challenge/validation.py) .
- **contributions.txt** has the details of contributions from each team member to the project.


## Usage

- Download `yelp_academic_dataset_business.json` from [Yelp Dataset Challenge](https://www.yelp.com/dataset_challenge/) and put this in the root folder
- At the root of repo, issue this command to set up the `PYTHONPATH`

    ```bash
    export PYTHONPATH="`pwd`/challenge:`pwd`/tests"
    ```
- Run the following commands at the root of repo


#### Run tests
```bash
python -m unittest validation_test
python -m unittest analysis_test
```

#### Run the classifier and print the classifier accuracy
```bash
python challenge/analysis.py
```
