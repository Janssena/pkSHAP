from sklearn.ensemble import RandomForestRegressor
from lib.colors import colors
from lib.metrics import mae
import pandas as pd
import numpy as np
import xgboost
import joblib
import shap
import os


def _get_groups(df): 
    return df.groupby(["ID", "OCC"] if "OCC" in df.keys() else ["ID"])


"""
Function to fit a RF or XGBoost model to data by means of a k-fold cross validation.
"""
def run_cv(df, param, covariates, k, model, should_save, save_dir):
    df_group = _get_groups(df)
    n = len(df_group)
    indexes = np.arange(0, n)
    np.random.shuffle(indexes)

    y = df.loc[:, param]

    train_error = np.zeros(k)
    test_error = np.zeros(k)
    models = []

    for i in range(0, k):
        print(colors.cyan + f"[Info] Running for fold {i + 1} ..." + colors.reset, end="\r" if i < k - 1 else colors.cyan + " DONE!" + colors.reset + "\n")
        start = round(i * n/k)
        end = round((i+1) * n/k) if i < (k - 1) else n
        test = indexes[start:end]
        train = np.delete(indexes, np.arange(start, end))

        m = xgboost.XGBRegressor() if model == 'xgboost' else RandomForestRegressor()
        m.fit(df.loc[train, covariates], y[train])

        models.append(m)
        train_error[i] = mae(y[train], m.predict(df.loc[train, covariates]))
        test_error[i] = mae(y[test], m.predict(df.loc[test, covariates]))

        if should_save:
            joblib.dump(m, os.path.join(save_dir, f"{param}_{model}_model_fold_{i + 1}.sav"))
            np.save(os.path.join(save_dir, f"{param}_train_set_fold_{i + 1}.npy"), train)

        np.save(os.path.join(save_dir, f"{param}_test_set_{i + 1}.npy"), test) # Should always be saved.

    if should_save:
        pd.DataFrame(np.stack([train_error, test_error], axis=1), columns=['train_error', 'test_error']).to_csv(os.path.join(save_dir, f"{param}_error.csv"))
    
    return {
        'models': models,
        'train_error': train_error, 
        'test_error': test_error
    }





def run_shap(df, covariates, param, models, save_dir):
    df_group = _get_groups(df)
    n = len(df_group)
    k = len(models)
    shap_values = np.zeros((k, n, len(covariates)))

    for (i, model) in enumerate(models):
        print(colors.cyan + f"[Info] Running SHAP analysis for fold {i + 1} ..." + colors.reset, end="\r" if i < k - 1 else colors.cyan + " DONE!" + colors.reset + "\n")
        test = np.load(os.path.join(save_dir, f"{param}_test_set_{i + 1}.npy"))
        expl = shap.TreeExplainer(model)
        shap_values[i, test, :] = expl.shap_values(df.loc[test, covariates])

    np.save(os.path.join(save_dir, f"{param}_shap_values.npy"), shap_values)

    return shap_values