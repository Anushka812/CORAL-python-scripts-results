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
The bacterial species is supplied as a command-line argument, allowing the same analytical workflow to be applied to multiple organism-specific datasets without modifying the script.
"""
if len(sys.argv) < 2:
    print(" Usage: python multivariate_regression1.py <bacteria_name>")
    print(" Example: python multivariate_regression1.py pseudomonas")
    sys.exit(1)
bacteria = sys.argv[1]

"""
2. Define input files and statistical thresholds
The script imports:
(i) the original patient-level dataset, and
(ii) the results of the preceding univariate regression analysis used to identify candidate predictors.
Predictors with univariate p-values below the predefined screening threshold are considered for multivariate modelling.
"""
univariate_dir = os.path.join("univariate_results", bacteria)
univariate_file = os.path.join(univariate_dir, f"univariate_outcome_results_{bacteria}.csv")
original_data_file = os.path.join("complete_data_116", f"{bacteria}.csv")

p_uni_threshold = 0.2      # Threshold to identify significant predictors in multivariate models.
p_multi_threshold = 0.05   # Threshold to identify significant multivariate associations.
print(f"\nBacteria: {bacteria}")
print(f"Univariate input: {univariate_file}")
print(f"Original data: {original_data_file}")

"""
3. Exclude mathematically dependent predictor–outcome pairs
Certain clinical variables represent related or derived measures (e.g., different mortality endpoints or hospital stay components). These variables are prevented from predicting one another to avoid modelling deterministic relationships that could produce misleading statistical associations.
"""

# Mortality-related outcomes
mortality_outcomes = [
    "Mortality in ICU",
    "Mortality in Hospital",
    "Mortality post 28 Day Discharge",
    "Overall Mortality"
]
# Hospital length-of-stay outcomes
stay_outcomes = [
    "ICU Stay in 28 Days",
    "GW Stay in 28 Days",
    "Total Hospital Stay in 28 Days"
]

# Construct a lookup table containing prohibited outcome–predictor combinations.
forbidden_pairs = set()
# Exclude mortality variables from predicting other mortality outcomes.
for outcome in mortality_outcomes:
    for predictor in mortality_outcomes:
        if predictor != outcome:
            forbidden_pairs.add((outcome, predictor))
# Exclude length-of-stay variables from predicting other length-of-stay outcomes.
for outcome in stay_outcomes:
    for predictor in stay_outcomes:
        if predictor != outcome:
            forbidden_pairs.add((outcome, predictor))
# Exclude mathematically related stay variables that represent overlapping measures.
for outcome in stay_outcomes:
    for predictor in stay_outcomes:
        if outcome != predictor:
            forbidden_pairs.add((outcome, predictor))
print(f"Created {len(forbidden_pairs)} forbidden predictor-outcome pairs")
print("Examples:", list(forbidden_pairs)[:5])

"""
4. Import input datasets
Variables are converted to numeric format where possible, and variables containing only missing values are excluded prior to statistical analysis.
"""
uni = pd.read_csv(univariate_file)
df = pd.read_csv(original_data_file, quotechar='"')
df = df.apply(pd.to_numeric, errors='coerce')
df = df.dropna(axis=1, how='all')

"""
5. Identify clinical outcomes available for analysis
"""
outcomes = uni['outcome'].unique()
print(f"\nOutcomes found: {outcomes}")

"""
6. Classify outcome variables according to data type
Outcome variables are classified as binary, ordinal or continuous so that the appropriate multivariate regression model can be selected automatically.
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
7. Perform multivariate regression analysis for each outcome
"""
all_results = []
for outcome in outcomes:
    print(f"\nProcessing outcome: {outcome}")
    # Select candidate predictors that satisfied the predefined univariate screening threshold.
    selected = uni[(uni['outcome'] == outcome) & (uni['p_value'] < p_uni_threshold)]['predictor'].tolist()
    selected = [p for p in selected if p != outcome]

    # Remove predictors that are mathematically related to the outcome under investigation.
    original_count = len(selected)
    selected = [p for p in selected if (outcome, p) not in forbidden_pairs]
    removed_count = original_count - len(selected)
    if removed_count > 0:
        print(f"  Removed {removed_count} forbidden predictors (mathematically related variables)")

    # Skip outcomes for which no eligible predictors remain after screening and exclusion.
    if len(selected) == 0:
        print(f"  No valid predictors after removing forbidden pairs. Skipping.")
        continue
    print(f"  Selected {len(selected)} valid predictors (univariate p < {p_uni_threshold})")
    # Assemble the analysis dataset by combining the selected predictors with the outcome
    # variable. Only complete cases are retained for multivariate modelling.
    y = df[outcome]
    X = df[selected]
    data = pd.concat([y, X], axis=1).dropna()
    # Skip model fitting when insufficient complete observations remain after listwise deletion.
    if data.shape[0] < 10:
        print(f"  Not enough observations ({data.shape[0]}) after listwise deletion. Skipping.")
        continue
    # Screen predictors for severe multicollinearity before regression modelling.
    X_temp = data[selected]
    if X_temp.shape[1] > 1:
        # Calculate pairwise correlations among candidate predictors.
        corr_matrix = X_temp.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
        if high_corr:
            print(f"  Warning: High correlation (>0.95) detected among: {high_corr}")

            # Remove redundant predictors to reduce instability arising from highly correlated
            # variables.
            selected = [p for p in selected if p not in high_corr[1:]]
            print(f"  Removed {len(high_corr)-1} predictors due to high correlation")
            X = df[selected]
            data = pd.concat([y, X], axis=1).dropna()
    # Prepare the final outcome and predictor matrices for multivariate regression analysis.
    y_clean = data[outcome]
    X_clean = data[selected]
    X_clean = sm.add_constant(X_clean)
    # Select the regression model according to the outcome type.
    y_type = var_type(y_clean)
    try:
        # Continuous outcomes are analysed using multiple linear regression.
        if y_type == 'continuous':
            model = sm.OLS(y_clean, X_clean).fit()
            coeffs = model.params
            pvals = model.pvalues
            ci = model.conf_int()
            reg_type = 'linear'
            # Record regression estimates and association inferential statistics for each model
            #parameter.
            for var in coeffs.index:
                all_results.append({
                    'outcome': outcome,
                    'predictor': var,
                    'beta_coefficient': coeffs[var],
                    'ci_lower': ci.loc[var, 0] if var in ci.index else np.nan,
                    'ci_upper': ci.loc[var, 1] if var in ci.index else np.nan,
                    'p_value': pvals[var],
                    'regression_type': reg_type,
                    'sample_size': data.shape[0],
                    'uni_p_threshold': p_uni_threshold
                })
        
       # Binary outcomes are analysed using multiple logistic regression.
        elif y_type == 'binary':
            model = sm.Logit(y_clean, X_clean).fit(disp=0, method='bfgs', maxiter=1000)
            coeffs = model.params
            pvals = model.pvalues
            ci = model.conf_int()
            reg_type = 'logistic'
            # Record regression estimates and associated statistics for each parameter.
            for var in coeffs.index:
                all_results.append({
                    'outcome': outcome,
                    'predictor': var,
                    'beta_coefficient': coeffs[var],
                    'ci_lower': ci.loc[var, 0] if var in ci.index else np.nan,
                    'ci_upper': ci.loc[var, 1] if var in ci.index else np.nan,
                    'p_value': pvals[var],
                    'regression_type': reg_type,
                    'sample_size': data.shape[0],
                    'uni_p_threshold': p_uni_threshold
                })

        # Ordinal outcomes are analysed using ordinal logistic regression.
        else:  
            y_cat = y_clean.astype('category')
            y_ord = y_cat.cat.codes
            X_no_const = X_clean.drop('const', axis=1)
            model = OrderedModel(y_ord, X_no_const, distr='logit')
            res = model.fit(method='bfgs', maxiter=1000, disp=0)
            coeffs = res.params
            pvals = 2 * (1 - norm.cdf(abs(coeffs / res.bse)))

            # Calculate Wald-based 95% confidence intervals.
            z = norm.ppf(0.975)
            ci_lower = coeffs - z * res.bse
            ci_upper = coeffs + z * res.bse
            reg_type = 'ordinal_logistic'
            # Record regression estimates and associated inferential statistics for each model
            # parameter.
            for var in coeffs.index:
                all_results.append({
                    'outcome': outcome,
                    'predictor': var,
                    'beta_coefficient': coeffs[var],
                    'ci_lower': ci_lower[var],
                    'ci_upper': ci_upper[var],
                    'p_value': pvals[var],
                    'regression_type': reg_type,
                    'sample_size': data.shape[0],
                    'uni_p_threshold': p_uni_threshold
                })
    # Continue analysing remaining outcomes if model fitting fails for an individual outcome.
    except Exception as e:
        print(f"  Error for outcome {outcome}: {e}")
        continue

"""
8. Identify statistically significant multivariate associations
A filtered results table is generated using the predefined multivariate significance threshold. The complete results are retained irrespective of statistical significance.
"""
result_df = pd.DataFrame(all_results)
filtered_df = result_df[result_df['p_value'] < p_multi_threshold].copy()

"""
9. Export multivariate regression results
Results are written to a bacteria-specific output directory. The complete multivariate results table is exported by default, while export of the filtered results table can be enabled if required.
"""
output_dir = os.path.join("multivariate_results", bacteria)
os.makedirs(output_dir, exist_ok=True)
full_output = os.path.join(output_dir, f"multivariate_full_puni{p_uni_threshold}_{bacteria}.csv")

# Uncomment below to export only statistically significant multivariate associations.
# filtered_output = os.path.join(output_dir, f"multivariate_filtered_pmulti{p_multi_threshold}_{bacteria}.csv")
result_df.to_csv(full_output, index=False)
# filtered_df.to_csv(filtered_output, index=False)
print(f"\n COMPLETED SUCCESSFULLY for {bacteria}")
print(f"  Results saved to: {full_output}")
