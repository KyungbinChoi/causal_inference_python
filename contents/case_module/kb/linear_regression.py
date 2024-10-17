"""
arguments
- p : input data path 
- y : outcome variable name
- t : treatment variable name
- c : control variable (Comma seperate)
"""
import os, argparse
import pandas as pd
import statsmodels.formula.api as smf

def get_args():
    parser = argparse.ArgumentParser(description="Perform OLS Linear Regression. Configure outcome/treatment/control variables.")
    parser.add_argument('-d', '--dataset', type=str, required=True, help='Path to the input dataset (CSV/PICKLE/PARQUET file)')
    parser.add_argument('-x', '--controls', type=str, required=False, help='Comma-separated column names for control variables (independent variables)')
    parser.add_argument('-t', '--treatment', type=str, required=True, help='Column name for the treatment variable')
    parser.add_argument('-o', '--outcome', type=str, required=True, help='Column name for the dependent variable')
    
    return parser.parse_args()

def ate_linear_regression(dataset_path, outcome_var, treatment_var, control_var):
    try:
        try:
            if dataset_path.split('.')[-1]=='csv':
                df = pd.read_csv(dataset_path)
            elif dataset_path.split('.')[-1]=='pkl':
                df = pd.read_pickle(dataset_path)
            elif dataset_path.split('.')[-1]=='parquet':
                df = pd.read_parquet(dataset_path)
        except Exception:
            print("Error: Not supported file format.")
            return
    except FileNotFoundError:
        print(f"Error: File '{dataset_path}' not found.")
        return
    
    control_var = control_var.split(',') if control_var else []

    for col in control_var + [treatment_var]:
        if df[col].dtype == 'object':
            df[col] = df[col].astype('category')

    if control_var:
        formula = f"{outcome_var} ~ " + ' + '.join([f"C({var})" if df[var].dtype.name == 'category' else var for var in control_var + [treatment_var]])
    else:
        formula = f"{outcome_var} ~ C({treatment_var})" if df[treatment_var].dtype.name == 'category' else f"{outcome_var} ~ {treatment_var}"

    print(formula)
    model = smf.ols(formula=formula, data=df).fit()

    print(model.summary())

    treatment_coef = model.params[treatment_var]
    treatment_pvalue = model.pvalues[treatment_var]
    treatment_conf_int = model.conf_int().loc[treatment_var]

    print(f"\nTreatment variable: '{treatment_var}'")
    print(f"Coefficient: {treatment_coef}")
    print(f"P-value: {treatment_pvalue}")
    print(f"95% Confidence Interval: {treatment_conf_int[0]}, {treatment_conf_int[1]}")

def main():
    args = get_args()
    ate_linear_regression(dataset_path= args.dataset,
                          outcome_var=args.outcome,
                          treatment_var=args.treatment,
                          control_var=args.controls)

if __name__ == "__main__":
    main()