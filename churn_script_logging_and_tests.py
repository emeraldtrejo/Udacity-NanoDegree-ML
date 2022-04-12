"""
Script contains test cases for churn_library.py
Author : Emerald Trejo
Date : April 11, 2022
"""

import os
import logging
from math import ceil
import churn_library as cls

logging.basicConfig(
    filename='./logs/churn_library.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

def test_import():
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        dataframe_test = cls.import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert dataframe_test.shape[0] > 0
        assert dataframe_test.shape[1] > 0
    except AssertionError as err:
        logging.error("Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda():
    '''
    test perform eda function
    '''
    dataframe_test_eda = cls.import_data("./data/bank_data.csv")
    try:
        cls.perform_eda(dataframe_test_eda)
        logging.info("Testing perform_eda: SUCCESS")
    #does column exist
    except KeyError as err:
        logging.error('Column "%s" not found', err.args[0])
        raise err
    #save figures to images folder
    #use asert from lesson #19
    #plt.savefig(fname='./images/churn.png')
    try:
        assert os.path.isfile("./images/churn.png") is True
        logging.info('File %s was found', 'churn.png')
    except AssertionError as err:
        logging.error('Not such file on disk')
        raise err
    #customer age cell image
    #plt.savefig(fname='./images/customer_age.png')
    try:
        assert os.path.isfile("./images/customer_age.png") is True
        logging.info('File %s was found', 'customer_age.png')
    except AssertionError as err:
        logging.error('Not such file on disk')
        raise err
    # marital status cell image
    #plt.savefig(fname='./images/marital_status.png')
    try:
        assert os.path.isfile("./images/marital_status.png") is True
        logging.info('File %s was found', 'marital_status.png')
    except AssertionError as err:
        logging.error('Not such file on disk')
        raise err
    # Total Transaction from cell
    #plt.savefig(fname='./images/total_transaction.png')
    try:
        assert os.path.isfile("./images/total_transaction.png") is True
        logging.info('File %s was found', 'total_transaction.png')
    except AssertionError as err:
        logging.error('Not such file on disk')
        raise err
    # Heatmap from cell
    #plt.savefig(fname='./images/heatmap.png')
    try:
        assert os.path.isfile("./images/heatmap.png") is True
        logging.info('File %s was found', 'heatmap.png')
    except AssertionError as err:
        logging.error('Not such file on disk')
        raise err


def test_encoder_helper():
    '''
    test encoder helper
    encoder helper: helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook
    '''
    # Load DataFrame
    df_encoder_helper = cls.import_data("./data/bank_data.csv")

    # Create `Churn` feature
    df_encoder_helper['Churn'] = df_encoder_helper['Attrition_Flag'].\
                                apply(lambda val: 0 if val=="Existing Customer" else 1)

    # Categorical Features
    cat_columns = ['Gender', 'Education_Level', 'Marital_Status',
                   'Income_Category', 'Card_Category']
    try:
        df_encoder_helper = cls.encoder_helper(
                            df_encoder_helper,
                            category_lst=[])

        # Data should be the same
        assert df_encoder_helper.equals(df_encoder_helper) is True
        logging.info("Testing encoder_helper(data_frame, category_lst=[]): SUCCESS")
    except AssertionError as err:
        logging.error("Testing encoder_helper(data_frame, category_lst=[]): ERROR")
        raise err

    try:
        df_encoder_helper = cls.encoder_helper(
                            df_encoder_helper,
                            category_lst=cat_columns)

        # Column names should be same
        assert df_encoder_helper.columns.equals(df_encoder_helper.columns) is True

        # Data should be different
        assert df_encoder_helper.equals(df_encoder_helper) is False
        logging.info(
            "Testing encoder_helper(data_frame, category_lst=cat_columns): SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper(data_frame, category_lst=cat_columns): ERROR")
        raise err

    try:
        df_encoder_helper = cls.encoder_helper(
                            df_encoder_helper,
                            category_lst=cat_columns)

        # Columns names should be different
        assert df_encoder_helper.columns.equals(df_encoder_helper.columns) is False
        # Data should be different
        assert df_encoder_helper.equals(df_encoder_helper) is False

        # Number of columns in encoded_df is the sum of columns in
        #data_frame and the newly created columns from cat_columns
        assert len(df_encoder_helper.columns) == len(df_encoder_helper.columns) + len(cat_columns)
        logging.info(
        "Testing encoder_helper(data_frame, category_lst=cat_columns): SUCCESS")
    except AssertionError as err:
        logging.error(
        "Testing encoder_helper(data_frame, category_lst=cat_columns): ERROR")
        raise err


def test_perform_feature_engineering():
    '''
    test perform_feature_engineering
    input dataframe, output testing and training data
    y = df['Churn']
    x_test = pd.DataFrame()
    '''
    # Load the DataFrame
    dataframe_pfe = cls.import_data("./data/bank_data.csv")

    #Churn feature
    dataframe_pfe['Churn'] = dataframe_pfe['Attrition_Flag'].\
        apply(lambda val: 0 if val=="Existing Customer" else 1)
    try:
        (_, x_test, _, _) = cls.perform_feature_engineering(dataframe_pfe)

        # `Churn` must be present in `data_frame` aka df
        assert 'Churn' in dataframe_pfe.columns
        logging.info("Testing perform_feature_engineering. `Churn` column exists: SUCCESS")
    except KeyError as err:
        logging.error('The term `Churn` column is not present: ERROR')
        raise err

    try:
        # X_test size should be 30% of `data_frame`
        assert (x_test.shape[0] == ceil(dataframe_pfe.shape[0]*0.3)) is True
        logging.info(
            'Testing perform_feature_engineering function. DataFrame sizes are consistent: SUCCESS')
    except AssertionError as err:
        logging.error(
            'Testing perform_feature_engineering function. DataFrame sizes are not correct: ERROR')
        raise err

def test_train_models():
    '''
    test train_models function, pass function as parameter
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
              checking to see if  all test and training data exists
              on the disk. This fucnction test will
              see if all test and train datas have been saved
              checking the functions: classification_report_image,
              feature_importance_plot, train_models
    '''
    # Load the DataFrame
    df_train_models = cls.import_data("./data/bank_data.csv")
    # Churn feature
    df_train_models['Churn'] = df_train_models['Attrition_Flag'].\
        apply(lambda val: 0 if val=="Existing Customer" else 1)

    # Feature engineering
    (x_train, x_test, y_train, y_test) = cls.perform_feature_engineering(
                                                    df_perform_feature_engineering \
                                                    =df_train_models)

    # Assert if `logistic_model.pkl` file is present for best models
    try:
        cls.train_models(x_train, x_test, y_train, y_test)
        assert os.path.isfile("./models/logistic_model.pkl") is True
        logging.info('File %s was found', 'logistic_model.pkl')
    except AssertionError as err:
        logging.error('No file exists on disk for logistic_model.pkl')
        raise err

    # Assert if `rfc_model.pkl` file is present for best models from train_models function
    try:
        assert os.path.isfile("./models/rfc_model.pkl") is True
        logging.info('File %s was found', 'rfc_model.pkl')
    except AssertionError as err:
        logging.error('No file exists on disk for rfc_model.pkl')
        raise err

    # Assert if `roc_curve_result.png` file is present from train_models function
    try:
        assert os.path.isfile('./images/results/roc_curve_result.png') is True
        logging.info('File %s was found', 'roc_curve_result.png')
    except AssertionError as err:
        logging.error('No file exists on disk for roc_curve_result.png')
        raise err

    # Assert if `randomforest_results.png` file is present
    try:
        assert os.path.isfile('./images/results/randomforest_results.png') is True
        logging.info('File %s was found', 'randomforest_results.png')
    except AssertionError as err:
        logging.error('No file exists on disk for randomforest_results.png')
        raise err

    # Assert if `logistic_results.png` file is present from
    #test train data from classification_report_image
    try:
        assert os.path.isfile('./images/results/logistic_results.png') is True
        logging.info('File %s was found', 'logistic_results.png')
    except AssertionError as err:
        logging.error('No file exists on disk for logistic_results.png')
        raise err

    # Assert if `feature_importances.png` file is present from feature_importance_plot
    try:
        assert os.path.isfile('./images/results/feature_importances.png') is True
        logging.info('File %s was found', 'feature_importances.png')
    except AssertionError as err:
        logging.error('No file exists on disk for feature_importances.png')
        raise err


if __name__ == "__main__":
    test_import()
    test_eda()
    test_encoder_helper()
    test_perform_feature_engineering()
    test_train_models()
    