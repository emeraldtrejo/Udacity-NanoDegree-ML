# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
Predict Customer Churn from data excel sheet. Create plots and test/train data for analysis.
Create churn_script_logging_and_test file to verify test and images have been saved on the disk.

## Running Files
How do you run your files? What should happen when you run your files?
Running the below code in the terminal should test each of the functions and provide any errors to a file stored in the logs folder.

ipython churn_script_logging_and_tests.py
You may choose to add an if __name__ == "__main__" block that allows you to run the code below and understand the results for each of the functions and refactored code associated with the original notebook.

ipython churn_library.py
You should check the pylint score using the below. Many of the common data science variable names don't meet pep8 standards, and these are probably okay to leave.

pylint churn_library.py
pylint churn_script_logging_and_tests.py
You should make sure you don't have any errors, and that your code is scoring as high as you can get it! You might shoot for well over 7.

To assist with meeting pep 8 guidelines, use autopep8 via the command line commands below:

autopep8 --in-place --aggressive --aggressive churn_script_logging_and_tests.py
autopep8 --in-place --aggressive --aggressive churn_library.py

