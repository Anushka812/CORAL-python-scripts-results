import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel
from scipy.stats import norm
import os
import sys
import warnings
warnings.filterwarnings('ignore')

"""
1. Specify the bacterial dataset to be analysed
The bacterial species is supplied as a command-line argument, allowing the same univariate regression workflow to be applied to multiple organism-specific datasets without modifying the
script.
"""

if len(sys.argv) < 2:
    print(" Usage: python univariate_regression1.py <bacteria_name>")
    print("   Example: python univariate_regression1.py pseudomonas")
    sys.exit(1)
bacteria = sys.argv[1]

"""
2. Import and prepare the patient-level dataset
The original dataset corresponding to the selected bacterial species is imported. Variables are converted to numeric format where possible, and variables containing only missing values are excluded before statistical analysis.
"""
file_path = os.path.join("complete_data_116", f"{bacteria}.csv")
print(f"\n Bacteria: {bacteria}")
print(f" Input file: {file_path}")
df = pd.read_csv(file_path, quotechar='"')

# Convert non-numeric entries to missing values to ensure compatibility with regression.
df = df.apply(pd.to_numeric, errors='coerce')
# Remove variables containing only missing observations.
df = df.dropna(axis=1, how='all')

"""
3. Define the clinical outcomes for univariate analysis. The analysis is performed separately for each predefined clinical outcome. Only outcomes present in the imported dataset are included.
"""
outcomes = [
    "Clinical Improvement",
    "ICU Stay in 28 Days",
    "GW Stay in 28 Days",
    "Total Hospital Stay in 28 Days",
    "Mortality in ICU",
    "Mortality in Hospital",
    "Mortality post 28 Day Discharge",
    "Overall Mortality",
    "Relapse post 28 Day Discharge",
    "Other Illness Leading to Hospitalization post 28 Days Discharge"
]

# Retain only outcome variables available in the dataset.
outcomes = [o for o in outcomes if o in df.columns]
print(f"Found {len(outcomes)} outcome variables: {outcomes}")

"""
4. Classify outcome variables according to data type. Outcome variables are classified as binary, ordinal or continuous so that the appropriate regression model can be selected automatically.
"""
def var_type(series):
    valid = series.dropna()
    if len(valid) == 0:
        return None
    unique = valid.unique()
    if len(unique) == 2 and set(unique).issubset({0, 1}):
        return 'binary'
    elif len(unique) <= 10:
        return 'ordinal'
    else:
        return 'continuous'

"""
5. Perform univariate regression analyses
Each clinical outcome is analysed independently against every eligible predictor variable. These analyses provide an initial assessment of individual predictor–outcome associations and
serve as a screening step for subsequent multivariate modelling.
"""

results = []
for out in outcomes:
    # Exclude missing outcome observations before model fitting.
    y = df[out].dropna()
    # Skip outcomes that contain insufficient variation for regression analysis.
    if y.nunique() < 2:
        print(f"Skipping {out}: less than 2 distinct values")
        continue
    y_type = var_type(y)
    print(f"\nProcessing outcome: {out} (type: {y_type})")

    # Evaluate every remaining variable as an individual predictor of the current outcome.
    predictors = [col for col in df.columns if col != out]
    for pred in predictors:
        x_raw = df[pred].dropna()
        # Skip predictors that contain insufficient variation.
        if x_raw.nunique() < 2:
            continue

        # Restrict the analysis to observations with complete data for both predictor and outcome.
        common_idx = y.index.intersection(x_raw.index)
        # Skip analyses with very small sample sizes.
        if len(common_idx) < 5:
            continue
        y_common = y.loc[common_idx]
        x_common = x_raw.loc[common_idx]
        X = sm.add_constant(x_common)
        try:
            # Continuous outcomes are analysed using simple linear regression.
            if y_type == 'continuous':
                model = sm.OLS(y_common, X).fit()
                beta = model.params[pred]
                pval = model.pvalues[pred]
                # Calculate the 95% confidence interval for the regression coefficient.
                ci = model.conf_int().loc[pred]
                ci_lower, ci_upper = ci[0], ci[1]
                reg_type = 'linear'
            # Binary outcomes are analysed using simple logistic regression.
            elif y_type == 'binary':
                model = sm.Logit(y_common, X).fit(disp=0, method='bfgs')
                beta = model.params[pred]
                pval = model.pvalues[pred]
                ci = model.conf_int().loc[pred]
                ci_lower, ci_upper = ci[0], ci[1]
                reg_type = 'logistic'
        # Ordinal outcomes are analysed using simple ordinal logistic regression.
            else:  
                y_cat = y_common.astype('category')
                y_ord = y_cat.cat.codes
                model = OrderedModel(y_ord, x_common, distr='logit')
                res = model.fit(method='nm', maxiter=500, disp=0)
                beta = res.params[0]
                pval = res.pvalues[0]
            # Calculate Wald-based 95% confidence intervals.
                se = res.bse[0]
                z = norm.ppf(0.975)
                ci_lower = beta - z * se
                ci_upper = beta + z * se
                reg_type = 'ordinal_logistic'

        # Record the regression coefficient and associated inferential statistics for the current pair.
            results.append({
                'outcome': out,
                'predictor': pred,
                'beta_coefficient': beta,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'p_value': pval,
                'regression_type': reg_type,
                # Flag predictors that satisfy the predefined screening threshold for consideration in the
                # subsequent multivariate analysis.
                'p_gt_0.2': pval < 0.2
        })

    # Continue analysing the remaining predictor–outcome pairs if the present model fails.
        except Exception as e:
            continue

"""
6. Export univariate regression results
Results are written to a bacteria-specific output directory and include all fitted univariate models. The output serves as the input for the subsequent multivariate regression script.
"""
output_dir = os.path.join("univariate_results", bacteria)
os.makedirs(output_dir, exist_ok=True)


output_file = os.path.join(output_dir, f"univariate_outcome_results_{bacteria}.csv")
result_df = pd.DataFrame(results)
result_df.to_csv(output_file, index=False)
print(f"\n Completed for {bacteria}. Saved {len(result_df)} regression results to: {output_file}")
