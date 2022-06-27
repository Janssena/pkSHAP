import argparse

def setup():
    parser = argparse.ArgumentParser(description='Script for running SHAP analysis on parameter estimates from base NLME model.')

    # Required arguments
    parser.add_argument('-f', '--file', dest='file', type=str, help='[Required] Raw NONMEM-style output file to run the analysis on. Should contain individual PK parameter estimates.')
    parser.add_argument('-p', '--parameter', dest='parameter', type=str, help='[Required] Variable detailing the column name of the parameter to run the SHAP analysis for. Should match one of the columns in FILE')

    # Optional arguments
    parser.add_argument('-c', '--covariates', nargs='+', dest='covariates', type=str, help='Covariates to include in the analysis.')
    parser.add_argument('-e', '--exclude', nargs='+', dest='exclude', type=str, help='Columns in the NONMEM-style output file to exclude from the analysis & visualizations.')

    # Optional arguments with defaults
    parser.add_argument('-k', dest='k', type=int, help='The number of folds to test on during the k-fold cross validation. Defaults to k = 5.', default=5)
    parser.add_argument('-m', '--model', dest='model', type=str, help='The model to use for the estimation of PK parameters. Should be one of ["rf" (default), "xgboost"].', default='rf')
    parser.add_argument('-s', '--save', '--save_result', dest='save', action='count', help='Boolean indicating whether to save all results of the analysis. By enabling this option the train/test set error, train/test set indexes, and the fitted models are also saved.', default=0)
    
    return parser.parse_args()

