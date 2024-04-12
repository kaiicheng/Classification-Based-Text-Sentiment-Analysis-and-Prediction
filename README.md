# Practical Applications in Machine Learning 

# Homework 3: Predicting Product Review Sentiment Using Classification

The goal of this assignment is to build a classification machine learning (ML) pipeline in a web application to use as a tool to analyze the models to gain useful insights about model performance. Using trained classification models, build a ML application that predicts whether a product review is positive or negative.

The <b>learning outcomes</b> for this assignment are:
* Preprocessing categorical data using n-gram, Term-Frequency Inverse document, and bag-of-word feature representations
* Build end-to-end classification pipeline with four classifiers 1) Logistic Regression and  2) Stochastic Gradient Descent.
* Evaluate classification methods using standard metrics including precision, recall, and accuracy, ROC Curves, and area under the curve.
* Develop a web application that walks users through steps of the classification pipeline and provide tools to analyze multiple methods across multiple metrics. 

## Assignment Outline
* Setup
* End-to-End Regression Models
* Testing Code with Github Autograder
* Reflection Assessment

# Setup

```
conda env create -f environment.yml
conda activate info5368 
```

# Reading Prerequisite 

* Review the jupyter notebook in Chapter 9 Unsupervised Learning of “Machine Géron, Aurélien. Hands-on machine learning with Scikit-Learn, Keras, and TensorFlow.” O’Reilly Media, Inc., 2022. Available on Canvas under ‘Library Reserves’.

# Amazon Products Dataset

This assignment involves training and evaluating ML end-to-end pipeline in a web application using the Amazon Product Reviews dataset. Millions of Amazon customers have contributed over a hundred million reviews to express opinions and describe their experiences regarding products on the Amazon.com website. This makes Amazon Customer Reviews a rich source of information for academic researchers in the fields of natural language processing (NLP), information retrieval (IR), and machine learning (ML), amongst others. Specifically, this dataset was constructed to represent a sample of customer evaluations and opinions, variation in the perception of a product across geographical regions, and promotional intent or bias in reviews. There are many features, but the <b>important</b> features include:
* name: name of Amazon product	
* reviews.text: text in review	
* reviews.title: title of reviews	

# Testing Code with Github Autograder

Test your homework solution as needed using Github Autograder. Clone your personal private copy of the homework assignment. Once the code is downloaded and stored on your computer in your directory of choice, you can start testing the assignment. To test your code, open a terminal and navigate to the directory of the homework3.py file. Then, type ‘pytest -v’ and press enter. The ‘-v’ stands for verbose which shows a summary of the passing and failing test cases.
```
pytest -v 
```

The autograder with print feedback to inform you what checkpoint functions are failing. Test homework3.py using this command in your terminal:
```
streamlit run homework3.py 
```

# Reflection Assessment

Complete on Canvas.





