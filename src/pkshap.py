from lib.helpers import run_cv, run_shap
import matplotlib.pyplot as plt
from datetime import datetime
from lib.colors import colors
from lib.args import setup
import pandas as pd
import numpy as np
import os

# Set up argparse:
args = setup()

def run():
    if not args.file:
        print(colors.cyan + "[Info] Please specify the NLME output file to run the analysis on as follows:" + colors.reset)
        print("$ python pkshap.py --file={enter filename here}")
        exit()
    elif not args.parameter:
        print(colors.cyan + "[Info] No parameter selected for the analysis, please supply one as follows:" + colors.reset)
        print("$ python pkshap.py --file=my_file.tab -p={enter PK parameter of interest here}")
        exit()
    
    df = pd.read_csv(args.file, header=1, sep='\s+')

    always_exclude = ["ID", "OCC", "TIME", "MDV", "DV", "EVID", "Y", "PRED", "IPRED", "RES", "WRES", "IWRES", "CWRES"]
    exclude = always_exclude + args.exclude if args.exclude is not None else always_exclude

    covariates = df.keys().to_list()
    for e in exclude:
        covariates.remove(e)

    dir_name = f'run_{datetime.now().strftime("%d-%m-%Y_%H:%M")}'

    try:
        os.mkdir(dir_name)
    except FileExistsError:
        print(colors.warn + "[Warning] Folder already exists, will save into existing folder" + colors.reset)
    except e:
        return e

    res = run_cv(df, args.parameter, covariates, args.k, args.model, args.save, dir_name)

    mean_train_err = np.mean(res['train_error'])
    mean_test_err = np.mean(res['test_error'])

    risk_overfit = (mean_test_err / mean_train_err) > 2.5
    shading = colors.warn if risk_overfit else colors.green

    print(f"Cross validation result: train set error: {mean_train_err:.5f} ± {(np.std(res['train_error'])):.3f},", end="")
    print(" test set error: " + shading + f"{mean_test_err:.5f} ± {(np.std(res['test_error'])):.3f}" + (" (>2.5x train error)" if risk_overfit else "") + colors.reset)

    run_shap(df, covariates, args.parameter, res['models'], dir_name)


if __name__ == "__main__":
    run()    
