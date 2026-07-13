import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel
from scipy.stats import norm
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# ----------------------------------------------------------------------
# 1. Get bacteria name from command line
# ----------------------------------------------------------------------
if len(sys.argv) < 2:
    print("❌ Usage: python univariate_regression1.py <bacteria_name>")
    print("   Example: python univariate_regression1.py pseudomonas")
    sys.exit(1)

bacteria = sys.argv[1]

# ----------------------------------------------------------------------
# 2. Load and clean data (dynamic path)
# ----------------------------------------------------------------------
file_path = os.path.join("complete_data_116", f"{bacteria}.csv")
print(f"\n📌 Bacteria: {bacteria}")
print(f"📌 Input file: {file_path}")

df = pd.read_csv(file_path, quotechar='"')

# Replace non‑numeric entries (e.g., "0,1") with NaN
df = df.apply(pd.to_numeric, errors='coerce')

# Drop columns that are completely empty
df = df.dropna(axis=1, how='all')

# ----------------------------------------------------------------------
# 3. Define the 10 outcome variables
# ----------------------------------------------------------------------
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

# Verify they exist
outcomes = [o for o in outcomes if o in df.columns]
print(f"Found {len(outcomes)} outcome variables: {outcomes}")

# ----------------------------------------------------------------------
# 4. Function to determine variable type
# ----------------------------------------------------------------------
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

# ----------------------------------------------------------------------
# 5. Run univariate regressions for each outcome vs all predictors
# ----------------------------------------------------------------------
results = []

for out in outcomes:
    y = df[out].dropna()
    if y.nunique() < 2:
        print(f"Skipping {out}: less than 2 distinct values")
        continue
    y_type = var_type(y)
    print(f"\nProcessing outcome: {out} (type: {y_type})")
    
    predictors = [col for col in df.columns if col != out]
    
    for pred in predictors:
        x_raw = df[pred].dropna()
        if x_raw.nunique() < 2:
            continue
        
        common_idx = y.index.intersection(x_raw.index)
        if len(common_idx) < 5:
            continue
        
        y_common = y.loc[common_idx]
        x_common = x_raw.loc[common_idx]
        X = sm.add_constant(x_common)
        
        try:
            if y_type == 'continuous':
                model = sm.OLS(y_common, X).fit()
                beta = model.params[pred]
                pval = model.pvalues[pred]
                # 95% confidence interval
                ci = model.conf_int().loc[pred]
                ci_lower, ci_upper = ci[0], ci[1]
                reg_type = 'linear'
            
            elif y_type == 'binary':
                model = sm.Logit(y_common, X).fit(disp=0, method='bfgs')
                beta = model.params[pred]
                pval = model.pvalues[pred]
                ci = model.conf_int().loc[pred]
                ci_lower, ci_upper = ci[0], ci[1]
                reg_type = 'logistic'
            
            else:  # ordinal
                y_cat = y_common.astype('category')
                y_ord = y_cat.cat.codes
                model = OrderedModel(y_ord, x_common, distr='logit')
                res = model.fit(method='nm', maxiter=500, disp=0)
                beta = res.params[0]
                pval = res.pvalues[0]
                # Compute CI from standard error (normal approximation)
                se = res.bse[0]
                z = norm.ppf(0.975)
                ci_lower = beta - z * se
                ci_upper = beta + z * se
                reg_type = 'ordinal_logistic'
            
            results.append({
                'outcome': out,
                'predictor': pred,
                'beta_coefficient': beta,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'p_value': pval,
                'regression_type': reg_type,
                'p_gt_0.2': pval < 0.2
            })
            
        except Exception as e:
            # Skip silently if model fails
            continue

# ----------------------------------------------------------------------
# 6. Save results to bacteria‑specific output folder
# ----------------------------------------------------------------------
output_dir = os.path.join("univariate_results", bacteria)
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, f"univariate_outcome_results_{bacteria}.csv")
result_df = pd.DataFrame(results)
result_df.to_csv(output_file, index=False)
print(f"\n✅ Completed for {bacteria}. Saved {len(result_df)} regression results to: {output_file}")