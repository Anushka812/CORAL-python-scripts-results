import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import warnings
warnings.filterwarnings('ignore')

"""
1. Specify the bacterial dataset to be analysed
The bacterial species is supplied as a command-line argument, allowing the same correlation analysis workflow to be applied to multiple organism-specific datasets without modifying the
script.
"""
if len(sys.argv) < 2:
    print(" Usage: python correlation1.py <bacteria_name>")
    print("   Example: python correlation1.py pseudomonas")
    sys.exit(1)
bacteria = sys.argv[1]

"""
2. Create a bacteria-specific output directory
All correlation results generated during the analysis are saved within an organism-specific directory to maintain separation of outputs between datasets.
"""

output_dir = os.path.join("correlation_results", bacteria)
os.makedirs(output_dir, exist_ok=True)

"""
3. Import and prepare the patient-level dataset
The original dataset corresponding to the selected bacterial species is imported. Variable names are standardised to ensure compatibility with downstream analyses, and variables are
converted to numeric format where appropriate.
"""

csv_file = os.path.join("complete_data_116", f"{bacteria}.csv")
print(f"\n Bacteria: {bacteria}")
print(f" Input file: {csv_file}")
if not os.path.exists(csv_file):
    print(f" Error: {csv_file} not found!")
    print(f" Current directory: {os.getcwd()}")
    exit()

data = pd.read_csv(csv_file)
# Standardise variable names by replacing non-alphanumeric characters with periods.
data.columns = data.columns.str.replace('[^0-9a-zA-Z]+', '.', regex=True)
data.columns = data.columns.str.replace('\\.+', '.', regex=True)

# Convert all variables except the patient identifier to numeric format where possible.
for col in data.columns:
    if col != 'Patient.ID':
        data[col] = pd.to_numeric(data[col], errors='coerce')
print(f" Loaded {data.shape[0]} rows, {data.shape[1]} columns")

"""
4. Classify variables according to measurement scale Variables are classified as binary, ordinal or continuous so that the appropriate correlation coefficient can be selected for each pairwise comparison.
"""

def get_variable_type(series):
    vals = series.dropna().unique()
    if len(vals) == 0:
        return "unknown"
    if set(vals).issubset({0, 1}) and len(vals) <= 2:
        return "binary"
    if len(vals) <= 5 and all(isinstance(v, (int, np.integer)) for v in vals):
        return "ordinal"
    return "continuous"

binary_vars = []
ordinal_vars = []
continuous_vars = []

for col in data.columns:
    if col == 'Patient.ID':
        continue
    var_type = get_variable_type(data[col])
    if var_type == "binary":
        binary_vars.append(col)
    elif var_type == "ordinal":
        ordinal_vars.append(col)
    elif var_type == "continuous":
        continuous_vars.append(col)

print(f"\n Variable types:")
print(f" -Binary: {len(binary_vars)} variables")
print(f" -Ordinal: {len(ordinal_vars)} variables")
print(f" -Continuous: {len(continuous_vars)} variables")

"""
5. Prepare variables for pairwise correlation analysis. All numeric variables, excluding the patient identifier, are retained for systematic pairwise correlation analysis.
"""

print("\n" + "-"*50)
print("CALCULATING CORRELATIONS")
print("-"*50)

# Select all numeric variables for correlation analysis.
numeric_cols = [col for col in data.columns if col != 'Patient.ID']
numeric_data = data[numeric_cols]

# Quantify pairwise associations among all eligible variables.
correlations = []
for i in range(len(numeric_cols)):
    for j in range(i+1, len(numeric_cols)):
        col1 = numeric_cols[i]
        col2 = numeric_cols[j]
        # Restrict the analysis to observations with complete data for both variables.
        temp_data = numeric_data[[col1, col2]].dropna()

        # Skip variable pairs with insufficient observations to estimate a correlation coefficient.
        if len(temp_data) < 3:
            continue

        # Calculate Spearman's rank correlation coefficient and its associated p-value.
        corr, pval = stats.spearmanr(temp_data[col1], temp_data[col2])

        # Classify the magnitude of the observed association according to predefined correlation
        # thresholds.
        abs_corr = abs(corr)
        if abs_corr >= 0.75:
            strength = "STRONG"
        elif abs_corr >= 0.50:
            strength = "MODERATE"
        else:
            strength = "WEAK"

        # Record the measurement scale of each variable to aid interpretation of the correlation.
        type1 = "binary" if col1 in binary_vars else ("ordinal" if col1 in ordinal_vars else "continuous")
        type2 = "binary" if col2 in binary_vars else ("ordinal" if col2 in ordinal_vars else "continuous")

        # Record the correlation coefficient together with the associated inferential statistics and 
        # descriptive classification.
        correlations.append({
            'Variable1': col1,
            'Variable1_Type': type1,
            'Variable2': col2,
            'Variable2_Type': type2,
            'Correlation': round(corr, 4),
            'P_Value': pval,
            'N': len(temp_data),
            'Strength': strength,
            'Direction': 'Positive' if corr > 0 else 'Negative',
            'Significant': pval < 0.05
        })

# Assemble and rank all pairwise correlations according to the absolute magnitude of the
# correlation coefficient.
corr_df = pd.DataFrame(correlations)
corr_df = corr_df.sort_values('Correlation', key=abs, ascending=False).reset_index(drop=True)
print(f" Calculated {len(corr_df)} correlation pairs")

"""
6. Export correlation analysis results
The complete set of pairwise correlation results is written to a bacteria-specific output directory. An optional filtered output containing only moderate and strong correlations can be generated if required.
"""

# Export the complete correlation results table.
corr_df.to_csv(os.path.join(output_dir, "correlations_all.csv"), index=False)
print(f" Saved: {output_dir}/correlations_all.csv")

# Uncomment below to export only moderate and strong
# correlations.
# important_corr = corr_df[corr_df['Strength'].isin(['STRONG', 'MODERATE'])]
# if len(important_corr) > 0:
#     important_corr.to_csv(os.path.join(output_dir, "correlations_main.csv"), index=False)
#     print(f" Saved: {output_dir}/correlations_main.csv ({len(important_corr)} pairs)")

print(f" ANALYSIS COMPLETE")
