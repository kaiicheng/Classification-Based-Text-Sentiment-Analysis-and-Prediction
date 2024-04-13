# Predicting Amazon Product Review Sentiment Using Classification

The goal of this project is to build a classification machine learning (ML) pipeline in a web application to use as a tool to analyze the models to gain useful insights about model performance. Using trained classification models, build a ML application that predicts whether a product review is positive or negative.


# Steps
- Preprocess textual data using techniques such as n-grams, Term-Frequency Inverse Document Frequency (TF-IDF), and bag-of-words feature representations.
- Build an end-to-end classification pipeline with two classifiers: 1) Logistic Regression and 2) Stochastic Gradient Descent (SGD).
- Evaluate classification methods using metrics like precision, recall, accuracy, ROC Curves, and area under the curve (AUC).
- Develop a web application that guides users through the steps of the classification pipeline and provides tools to analyze and compare methods across multiple metrics.


# Setup

```
conda env create -f environment.yml
conda activate info5368 
```

# Amazon Products Dataset

This project involves training and evaluating ML end-to-end pipeline in a web application using the Amazon Product Reviews dataset. Millions of Amazon customers have contributed over a hundred million reviews to express opinions and describe their experiences regarding products on the Amazon.com website. This makes Amazon Customer Reviews a rich source of information for academic researchers in the fields of natural language processing (NLP), information retrieval (IR), and machine learning (ML), amongst others. Specifically, this dataset was constructed to represent a sample of customer evaluations and opinions, variation in the perception of a product across geographical regions, and promotional intent or bias in reviews. There are many features, but the <b>important</b> features include:
* name: name of Amazon product	
* reviews.text: text in review	
* reviews.title: title of reviews	

# Testing Code with Github Autograder

Type ‘pytest -v’ and press enter. The ‘-v’ stands for verbose which shows a summary of the passing and failing test cases.
```
pytest -v 
```

The autograder with print feedback to inform you what checkpoint functions are failing. Test homework3.py using this command in your terminal:
```
streamlit run homework3.py 
```
