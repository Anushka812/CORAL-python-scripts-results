import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel
from scipy.stats import norm
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# ------------------------------------------------------------
# 1. Get bacteria name from command line
# ------------------------------------------------------------
if len(sys.argv) < 2:
    print("❌ Usage: python multivariate_regression1.py <bacteria_name>")
    print("   Example: python multivariate_regression1.py pseudomonas")
    sys.exit(1)

bacteria = sys.argv[1]

# ------------------------------------------------------------
# 2. File paths and thresholds (dynamic per bacteria)
# ------------------------------------------------------------
univariate_dir = os.path.join("univariate_results", bacteria)
univariate_file = os.path.join(univariate_dir, f"univariate_outcome_results_{bacteria}.csv")
original_data_file = os.path.join("complete_data_116", f"{bacteria}.csv")

p_uni_threshold = 0.2      # p-value threshold to select predictors from univariate results
p_multi_threshold = 0.05   # only keep predictors with multivariate p-value < this in final output

print(f"\n📌 Bacteria: {bacteria}")
print(f"📌 Univariate input: {univariate_file}")
print(f"📌 Original data: {original_data_file}")

# ------------------------------------------------------------
# 3. Define forbidden predictor-outcome pairs
#    These are mathematically related variables that should NEVER predict each other
# ------------------------------------------------------------
# Group 1: Mortality outcomes (these often perfectly predict each other)
mortality_outcomes = [
    "Mortality in ICU",
    "Mortality in Hospital", 
    "Mortality post 28 Day Discharge",
    "Overall Mortality"
]

# Group 2: Stay duration outcomes (these are mathematically additive)
stay_outcomes = [
    "ICU Stay in 28 Days",
    "GW Stay in 28 Days",
    "Total Hospital Stay in 28 Days"
]

# Create a set of forbidden (outcome, predictor) pairs
# Also forbid predictors that are in the same mathematical family as the outcome
forbidden_pairs = set()

# For any mortality outcome, forbid all other mortality variables as predictors
for outcome in mortality_outcomes:
    for predictor in mortality_outcomes:
        if predictor != outcome:
            forbidden_pairs.add((outcome, predictor))

# For any stay outcome, forbid all other stay variables as predictors
for outcome in stay_outcomes:
    for predictor in stay_outcomes:
        if predictor != outcome:
            forbidden_pairs.add((outcome, predictor))

# Also forbid predictors that are mathematical combinations
# Total Hospital Stay should not predict ICU Stay or GW Stay (and vice versa)
for outcome in stay_outcomes:
    for predictor in stay_outcomes:
        if outcome != predictor:
            forbidden_pairs.add((outcome, predictor))

print(f"Created {len(forbidden_pairs)} forbidden predictor-outcome pairs")
print("Examples:", list(forbidden_pairs)[:5])

# ------------------------------------------------------------
# 4. Read data
# ------------------------------------------------------------
uni = pd.read_csv(univariate_file)
df = pd.read_csv(original_data_file, quotechar='"')
df = df.apply(pd.to_numeric, errors='coerce')
df = df.dropna(axis=1, how='all')

# ------------------------------------------------------------
# 5. List of outcomes
# ------------------------------------------------------------
outcomes = uni['outcome'].unique()
print(f"\nOutcomes found: {outcomes}")

# ------------------------------------------------------------
# 6. Function to determine variable type (binary/ordinal/continuous)
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# 7. Multivariate analysis for each outcome
# ------------------------------------------------------------
all_results = []

for outcome in outcomes:
    print(f"\nProcessing outcome: {outcome}")
    
    # Select predictors with univariate p < threshold
    selected = uni[(uni['outcome'] == outcome) & (uni['p_value'] < p_uni_threshold)]['predictor'].tolist()
    selected = [p for p in selected if p != outcome]
    
    # Remove forbidden predictors
    original_count = len(selected)
    selected = [p for p in selected if (outcome, p) not in forbidden_pairs]
    removed_count = original_count - len(selected)
    
    if removed_count > 0:
        print(f"  Removed {removed_count} forbidden predictors (mathematically related variables)")
    
    if len(selected) == 0:
        print(f"  No valid predictors after removing forbidden pairs. Skipping.")
        continue
    
    print(f"  Selected {len(selected)} valid predictors (univariate p < {p_uni_threshold})")
    
    # Prepare data from original dataframe
    y = df[outcome]
    X = df[selected]
    data = pd.concat([y, X], axis=1).dropna()
    
    if data.shape[0] < 10:
        print(f"  Not enough observations ({data.shape[0]}) after listwise deletion. Skipping.")
        continue
    
    # Check for perfect multicollinearity
    X_temp = data[selected]
    if X_temp.shape[1] > 1:
        # Check correlation matrix
        corr_matrix = X_temp.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
        if high_corr:
            print(f"  Warning: High correlation (>0.95) detected among: {high_corr}")
            # Remove one of the highly correlated predictors (keep first)
            selected = [p for p in selected if p not in high_corr[1:]]
            print(f"  Removed {len(high_corr)-1} predictors due to high correlation")
            X = df[selected]
            data = pd.concat([y, X], axis=1).dropna()
    
    y_clean = data[outcome]
    X_clean = data[selected]
    X_clean = sm.add_constant(X_clean)
    
    y_type = var_type(y_clean)
    
    try:
        if y_type == 'continuous':
            model = sm.OLS(y_clean, X_clean).fit()
            coeffs = model.params
            pvals = model.pvalues
            ci = model.conf_int()
            reg_type = 'linear'
            
            # Store results
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
            
        elif y_type == 'binary':
            model = sm.Logit(y_clean, X_clean).fit(disp=0, method='bfgs', maxiter=1000)
            coeffs = model.params
            pvals = model.pvalues
            ci = model.conf_int()
            reg_type = 'logistic'
            
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
            
        else:  # ordinal
            y_cat = y_clean.astype('category')
            y_ord = y_cat.cat.codes
            X_no_const = X_clean.drop('const', axis=1)
            model = OrderedModel(y_ord, X_no_const, distr='logit')
            res = model.fit(method='bfgs', maxiter=1000, disp=0)
            coeffs = res.params
            pvals = 2 * (1 - norm.cdf(abs(coeffs / res.bse)))
            z = norm.ppf(0.975)
            ci_lower = coeffs - z * res.bse
            ci_upper = coeffs + z * res.bse
            reg_type = 'ordinal_logistic'
            
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
            
    except Exception as e:
        print(f"  Error for outcome {outcome}: {e}")
        continue

# ------------------------------------------------------------
# 8. Filter to keep only predictors with multivariate p < p_multi_threshold
# ------------------------------------------------------------
result_df = pd.DataFrame(all_results)
filtered_df = result_df[result_df['p_value'] < p_multi_threshold].copy()

# ------------------------------------------------------------
# 9. Save both full and filtered results to bacteria-specific output folder
# ------------------------------------------------------------
output_dir = os.path.join("multivariate_results", bacteria)
os.makedirs(output_dir, exist_ok=True)

full_output = os.path.join(output_dir, f"multivariate_full_puni{p_uni_threshold}_{bacteria}.csv")
# Uncomment below if you also want to save the filtered results
# filtered_output = os.path.join(output_dir, f"multivariate_filtered_pmulti{p_multi_threshold}_{bacteria}.csv")

result_df.to_csv(full_output, index=False)
# filtered_df.to_csv(filtered_output, index=False)

print(f"\n✅ COMPLETED SUCCESSFULLY for {bacteria}")
print(f"   Results saved to: {full_output}")