"""
Script contains functions to execute churn data.
Validation tests are performed with churn_script_logging_and_tests.py
Author : Emerald Trejo
Date : April 11, 2022
"""
# import libraries
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_roc_curve, classification_report
os.environ['QT_QPA_PLATFORM']='offscreen'

def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    df1 = pd.read_csv(pth)
    df1['Churn'] = df1['Attrition_Flag'].apply(lambda x: 0 if x =='Existing Customer' else 1)
    return df1


def perform_eda(dataframe_perform_eda):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    #saving images from churnnotebook, get images from churn notbook by copying the code

    #churn cell
    plt.figure(figsize=(20, 10))
    #make a histogram
    dataframe_perform_eda['Churn'].hist()
    plt.savefig(fname='./images/eda/churn.png')

    #customer age cell image
    plt.figure(figsize=(20, 10))
    dataframe_perform_eda['Customer_Age'].hist()
    plt.savefig(fname='./images/eda/customer_age.png')

   # marital status cell image
    plt.figure(figsize=(20, 10))
    #normalize is fraction counts
    dataframe_perform_eda.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig(fname='./images/eda/marital_status.png')

        # Total Transaction from cell
    plt.figure(figsize=(20, 10))
    sns.distplot(dataframe_perform_eda['Total_Trans_Ct'],kde=True)
    plt.savefig(fname='./images/eda/total_transaction.png')

        # Heatmap from cell
    plt.figure(figsize=(20, 10))
    sns.heatmap(dataframe_perform_eda.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig(fname='./images/eda/heatmap.png')


def encoder_helper(df_encoder_helper, category_lst):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming
            variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    #extract categories from csv that can be used for data analysis
    #copy same format from cell 15 for churn word
    #genderGrp = df.groupby("Gender").mean()['Churn'].to_dict()
    column_lst = []
    for category in category_lst:
        column_lst.append(f'{category}_mean')
        #churn_lst = df.groupby(category).mean()['Churn']
        churn_lst = df.groupby(category).mean()['Churn'].to_dict()
        df_encoder_helper[f'{category}_Churn'] = df_encoder_helper[category] \
        .apply(lambda x: churn_lst[x])
    return df_encoder_helper, column_lst

def perform_feature_engineering(df_perform_feature_engineering):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could
              be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    #copied from notebook
    keep_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book',
             'Total_Relationship_Count', 'Months_Inactive_12_mon',
             'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
             'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
             'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
             'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
             'Income_Category_Churn', 'Card_Category_Churn']
    x_pfe = df_perform_feature_engineering[keep_cols]
        #copied from notebook
    y_pfe = df_perform_feature_engineering['Churn']
        #reference: https://realpython.com/train-test-split-python-data/
        # train test split
    x_train_pfe, x_test_pfe, y_train, y_test = train_test_split(x_pfe, \
        y_pfe, test_size= 0.3, random_state=42)
    return (x_train_pfe, x_test_pfe, y_train, y_test)

def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    #copied from cells for Logisitic Regression Train
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01,1.25,str('Logistic Regression Train'), \
             {'fontsize': 10},fontproperties = 'monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, \
            y_train_preds_lr)),{'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.6, str('Logistic Regression Test'), \
             {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, \
     y_test_preds_lr)),{'fontsize': 10},fontproperties = 'monospace')
    plt.axis('off')
    plt.savefig(fname='./images/results/logistic_results.png')
    #copied from cells for random forest
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Random Forest Train'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train,
        y_train_preds_rf)), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.6, str('Random Forest Test'), \
        {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, \
        y_test_preds_rf)), {'fontsize': 10}, fontproperties = 'monospace')
    plt.axis('off')
    plt.savefig(fname='./images/results/randomforest_results.png')


def feature_importance_plot(model, x_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Feature importances
    importances = model.best_estimator_.feature_importances_

    # Sort Feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Sorted feature importances
    names = [x_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(25, 15))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(x_data.shape[1]), importances[indices])

    # x-axis labels
    plt.xticks(range(x_data.shape[1]), names, rotation=90)

    # Save the image
    plt.savefig(fname=output_pth + 'feature_importances.png')

def train_models(x_train, x_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    #train
    # train test split
    #train data has features
    #take 70% of data for training, 30% for test set
    # grid search
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression()
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth' : [4,5,100],
        'criterion' :['gini', 'entropy']
    }
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(x_train, y_train)
    lrc.fit(x_train, y_train)
    #save, copied from notebook
    # save best model
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')
    # Compute train and test predictions for RandomForestClassifier
    y_train_preds_rf = cv_rfc.best_estimator_.predict(x_train)
    y_test_preds_rf  = cv_rfc.best_estimator_.predict(x_test)

    # Compute train and test predictions for LogisticRegression
    y_train_preds_lr = lrc.predict(x_train)
    y_test_preds_lr  = lrc.predict(x_test)

    # Compute ROC curve
    plt.figure(figsize=(15, 8))
    axis = plt.gca()
    lrc_plot = plot_roc_curve(lrc, x_test, y_test, ax=axis, alpha=0.8)
    rfc_disp = plot_roc_curve(cv_rfc.best_estimator_, x_test, y_test, ax=axis, alpha=0.8)
    plt.savefig(fname='./images/results/roc_curve_result.png')
    plt.show()
    # Compute and results
    classification_report_image(y_train, y_test,
                                y_train_preds_lr, y_train_preds_rf,
                                y_test_preds_lr,  y_test_preds_rf)

    # Compute and feature importance
    feature_importance_plot(model=cv_rfc,
                            x_data=x_test,
                            output_pth='./images/results/')
if __name__ == "__main__":
    df = import_data(r"./data/bank_data.csv")
    perform_eda(df)
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category']
    quant_columns = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio']
    df, copied_column_lst = encoder_helper(df, cat_columns)
    print(copied_column_lst)
    print(df.head())
    (X_train_results,X_test_results,y_train_results, \
    y_test_results) = perform_feature_engineering(df)
    train_models(X_train_results, X_test_results, y_train_results,y_test_results)
   