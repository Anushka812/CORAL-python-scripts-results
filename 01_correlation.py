import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# =====================================================================
# STEP 1: GET BACTERIA NAME FROM COMMAND LINE
# =====================================================================

if len(sys.argv) < 2:
    print("❌ Usage: python correlation1.py <bacteria_name>")
    print("   Example: python correlation1.py pseudomonas")
    sys.exit(1)

bacteria = sys.argv[1]

# =====================================================================
# STEP 2: CREATE OUTPUT DIRECTORY (per bacteria)
# =====================================================================

output_dir = os.path.join("correlation_results", bacteria)
os.makedirs(output_dir, exist_ok=True)

# =====================================================================
# STEP 3: LOAD YOUR DATA
# =====================================================================

csv_file = os.path.join("complete_data_116", f"{bacteria}.csv")
print(f"\n📌 Bacteria: {bacteria}")
print(f"📌 Input file: {csv_file}")

if not os.path.exists(csv_file):
    print(f"❌ Error: {csv_file} not found!")
    print(f"   Current directory: {os.getcwd()}")
    exit()

data = pd.read_csv(csv_file)
data.columns = data.columns.str.replace('[^0-9a-zA-Z]+', '.', regex=True)
data.columns = data.columns.str.replace('\\.+', '.', regex=True)

for col in data.columns:
    if col != 'Patient.ID':
        data[col] = pd.to_numeric(data[col], errors='coerce')

print(f"✅ Loaded {data.shape[0]} rows, {data.shape[1]} columns")

# =====================================================================
# STEP 4: IDENTIFY VARIABLE TYPES
# =====================================================================

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

print(f"\n📊 Variable types:")
print(f"  • Binary: {len(binary_vars)} variables")
print(f"  • Ordinal: {len(ordinal_vars)} variables")
print(f"  • Continuous: {len(continuous_vars)} variables")

# =====================================================================
# STEP 5: CALCULATE CORRELATIONS
# =====================================================================

print("\n" + "-"*50)
print("CALCULATING CORRELATIONS")
print("-"*50)

# Get all numeric columns
numeric_cols = [col for col in data.columns if col != 'Patient.ID']
numeric_data = data[numeric_cols]

# Calculate correlations
correlations = []
for i in range(len(numeric_cols)):
    for j in range(i+1, len(numeric_cols)):
        col1 = numeric_cols[i]
        col2 = numeric_cols[j]
        
        temp_data = numeric_data[[col1, col2]].dropna()
        if len(temp_data) < 3:
            continue
        
        corr, pval = stats.spearmanr(temp_data[col1], temp_data[col2])
        
        abs_corr = abs(corr)
        if abs_corr >= 0.75:
            strength = "STRONG"
        elif abs_corr >= 0.50:
            strength = "MODERATE"
        else:
            strength = "WEAK"
        
        type1 = "binary" if col1 in binary_vars else ("ordinal" if col1 in ordinal_vars else "continuous")
        type2 = "binary" if col2 in binary_vars else ("ordinal" if col2 in ordinal_vars else "continuous")
        
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

corr_df = pd.DataFrame(correlations)
corr_df = corr_df.sort_values('Correlation', key=abs, ascending=False).reset_index(drop=True)

print(f"✅ Calculated {len(corr_df)} correlation pairs")

# =====================================================================
# STEP 6: SAVE ALL CORRELATION RESULTS TO OUTPUT DIRECTORY
# =====================================================================

# Save all correlations
corr_df.to_csv(os.path.join(output_dir, "correlations_all.csv"), index=False)
print(f"✅ Saved: {output_dir}/correlations_all.csv")

# (Optional) Uncomment below to save only strong/moderate correlations
# important_corr = corr_df[corr_df['Strength'].isin(['STRONG', 'MODERATE'])]
# if len(important_corr) > 0:
#     important_corr.to_csv(os.path.join(output_dir, "correlations_main.csv"), index=False)
#     print(f"✅ Saved: {output_dir}/correlations_main.csv ({len(important_corr)} pairs)")

print("✅ ANALYSIS COMPLETE")