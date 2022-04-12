# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
In this project, we will identify credit card customers that are most likely to stop being a customer. The Project will include PEP8 Python package that follows coding best practices for implementing software. The package will also have the flexibility of being run interactively or from the command-line interface (CLI).

## Files in the Repo
data folder contains the bank_data.csv
Images folder contains eda and results folder which contain images created from running the code
logsfolder contains the churn_library.log file
models folder contains the logisitic & rfc models

Laying on the root folder is the churn_library.py, churn_script_logging_and_tests.py, Guide.ipynb, README.md, churn_notebook.ipynb

## Running Files
Install the linter and auto-formatter: 

Run the folllowing command on the CLI to install the pylint software:

pip install pylint 

Run the following command on the CLI to install autopep8. 
Autopep8 automatically formats Python code to conform to the PEP 8 style guide.:

pip install autopep8

To run your programs, here are the lines:
python churn_library.py 
python python_script_logging_and_tests.py

To check the pylint score using the below: 
pylint churn_library.py 
pylint churn_script_logging_and_tests.py

To assist with meeting pep 8 guidelines, use autopep8 via the command line commands below: autopep8 --in-place --aggressive --aggressive churn_script_logging_and_tests.py autopep8 --in-place --aggressive --aggressive churn_library.py

## Required Libraries to successfully run this project:
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import plot_roc_curve, classification_report