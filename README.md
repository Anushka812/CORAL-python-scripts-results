# CORAL: Clinical Outcomes Regression and Analytical Learning Pipeline

**CORAL** is a fully automated, modular pipeline for high‑throughput correlation, univariate, and multivariate regression analyses across multiple bacterial species. Designed for clinical and microbiological researchers, it processes a folder of CSV files (one per organism) and outputs structured results with minimal user intervention—no coding experience required, yet fully customizable for advanced users.

---

## 📌 What This Pipeline Does

CORAL performs three classical statistical analyses for each bacterial dataset:

| Analysis | Description |
|----------|-------------|
| **Correlation Analysis** | Pairwise Spearman rank‑order correlations between all numeric variables, with strength classification (Strong/Moderate/Weak) and statistical significance. |
| **Univariate Regression** | For each of 10 predefined clinical outcomes, every other variable is tested as a sole predictor using the appropriate regression model (linear, logistic, or ordinal) based on the outcome's data type. |
| **Multivariate Regression** | Predictors with univariate *p* < 0.20 are considered for a combined model. Forbidden pairs (e.g., mathematically related variables) are excluded, collinearity is checked, and only predictors with *p* < 0.05 are retained in the final model. |

All results are saved in organised subfolders, facilitating cross‑bacteria comparisons and downstream meta‑analyses.

---

## 🧠 Statistical & Methodological Details

### Variable Type Detection
The pipeline automatically classifies each column (excluding `Patient.ID`) into one of three types:

- **Binary**: Exactly two unique values, both in {0, 1}.
- **Ordinal**: ≤10 unique integer values (e.g., Likert scales, severity scores).
- **Continuous**: More than 10 unique values, or non‑integer values (treated as numeric).

This classification drives the choice of regression model in both univariate and multivariate steps.

### Correlation Analysis
- **Method**: Spearman's rank correlation (non‑parametric, robust to outliers and non‑linear monotonic relationships).
- **Output**: Correlation coefficient (ρ), *p*-value, and strength label:
  - **Strong**: |ρ| ≥ 0.75
  - **Moderate**: 0.50 ≤ |ρ| < 0.75
  - **Weak**: |ρ| < 0.50
- **Implementation**: `scipy.stats.spearmanr`; pairs with <3 complete observations are skipped.

### Univariate Regression
For each outcome–predictor pair, the pipeline selects the regression model based on the outcome's type:

| Outcome Type | Model | Implementation |
|--------------|-------|----------------|
| Continuous | Ordinary Least Squares (OLS) linear regression | `statsmodels.api.OLS` |
| Binary | Logistic regression | `statsmodels.api.Logit` (BFGS optimizer) |
| Ordinal | Ordinal logistic regression | `statsmodels.miscmodels.ordinal_model.OrderedModel` |

**Outputs per pair**: Coefficient (beta), *p*-value, 95% confidence interval, and regression type.

### Multivariate Regression
1. **Predictor Selection**: All predictors with univariate *p* < 0.20 are candidates.
2. **Forbidden Pair Exclusion**: Prevents mathematically redundant variables from predicting each other:
   - **Mortality outcomes**: `Mortality in ICU`, `Mortality in Hospital`, `Mortality post 28 Day Discharge`, `Overall Mortality` — these are mutually forbidden as predictors.
   - **Stay outcomes**: `ICU Stay in 28 Days`, `GW Stay in 28 Days`, `Total Hospital Stay in 28 Days` — these are mutually forbidden.
3. **Collinearity Check**: (Implemented in the code; ensures the predictor set is not perfectly multicollinear.)
4. **Final Model**: Only predictors with multivariate *p* < 0.05 are retained in the output.

---

## 🧩 Pipeline Workflow (Per Bacterium)

1. **Read the data** – Loads your clinical spreadsheet.
2. **Identify variable types** – Automatically detects binary, ordinal, and continuous variables.
3. **Correct ordinal direction** – If an ordinal scale is coded in reverse (e.g., higher numbers mean better, but clinically they should mean worse), the pipeline flips it to ensure correct interpretation.
4. **Calculate all pairwise Spearman correlations** – Saves the correlation coefficient, *p*-value, and strength.
5. **Run univariate regressions** – For 10 predefined clinical outcomes, tests every other variable as a predictor, choosing the correct regression model based on the outcome type.
6. **Run multivariate regressions** – Selects predictors with *p* < 0.20 in the univariate step, removes redundant/forbidden pairs, checks for collinearity, and fits a multivariate model. Only predictors with *p* < 0.05 are kept.

---

## 🖥️ Who Can Use This?

| Audience | How They Benefit |
|----------|------------------|
| **Clinicians / Medical Doctors** | No coding needed; follow the setup and run one command. |
| **Biostatisticians** | Easy to inspect, modify thresholds (e.g., *p*-value cutoffs), or extend with additional models. |
| **Bioinformaticians** | Fully scripted for reproducibility, batch processing, and integration into larger workflows. |

---

## 📦 System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **OS** | Windows 10/11, macOS (Intel or Apple Silicon), or Linux | Same |
| **RAM** | 4 GB | 8 GB (for large datasets) |
| **Storage** | ~100 MB (pipeline) + data + results | Depends on dataset size |
| **Python** | 3.9 or later | 3.10+ |

---

## 🔧 Step‑by‑Step Setup Guide

### 1. Install Python

**Windows**
- Download from [python.org](https://www.python.org/downloads/) (version 3.9 or later).
- **During installation, check “Add Python to PATH”**.
- Verify: Open Command Prompt and run:
  ```bash
  python --version
  ```

**macOS**
- Python is usually pre‑installed, but we recommend installing a fresh version via Homebrew:
  ```bash
  brew install python
  ```
  (If you don't have Homebrew, install it from [brew.sh](https://brew.sh/).)

**Linux**
- Use your package manager (e.g., `sudo apt install python3 python3-pip` on Ubuntu/Debian).

---

### 2. Download the Pipeline

- Visit the GitHub repository: [https://github.com/Anushka812/CORAL-python-scripts-results](https://github.com/Anushka812/CORAL-python-scripts-results)
- Click **Code → Download ZIP**.
- Extract the ZIP to a folder (e.g., `C:\pipeline` on Windows, or `~/pipeline` on macOS/Linux).
- Open your terminal/command prompt and navigate to that folder:
  ```bash
  cd path/to/pipeline
  ```

---

### 3. Create a Virtual Environment (Recommended)

A virtual environment isolates the required Python packages, preventing conflicts with other projects.

**Windows**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux**
```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` appear at the start of your command prompt.

---

### 4. Install Required Python Packages

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note for Apple Silicon (M1/M2/M3) Mac users**: If you encounter issues with `statsmodels`, try:
> ```bash
> pip install numpy scipy pandas matplotlib seaborn statsmodels --force-reinstall --no-cache-dir
> ```

If a `requirements.txt` file is not present, install the core dependencies manually:
```bash
pip install pandas numpy scipy statsmodels matplotlib seaborn
```

---

### 5. Prepare Your Data

- Place all bacterial CSV files inside the folder **`complete_data_116/`** (this folder already exists in the pipeline).
- **Naming**: Each CSV file must be named exactly as the bacterium (e.g., `pseudomonas.csv`, `ecoli.csv`). The name (without `.csv`) is used as the bacteria identifier throughout the pipeline.
- **Format**:
  - Must contain a `Patient.ID` column (string or numeric, used as an index).
  - All other columns should be **numeric** (integers or floats). Non‑numeric values will be coerced to `NaN`.
  - Column names should not contain special characters (the pipeline replaces non‑alphanumeric characters with dots).
- **Outcomes**: The pipeline expects the following 10 outcome columns (exact names):
  - `Clinical Improvement`
  - `ICU Stay in 28 Days`
  - `GW Stay in 28 Days`
  - `Total Hospital Stay in 28 Days`
  - `Mortality in ICU`
  - `Mortality in Hospital`
  - `Mortality post 28 Day Discharge`
  - `Overall Mortality`
  - `Relapse post 28 Day Discharge`
  - `Other Illness Leading to Hospitalization post 28 Days Discharge`

  If your column names differ, edit the `outcomes` list inside `02_univariate_regression.py` and `03_multivariate_regression.py`.

---

## 🚀 Running the Pipeline

Once everything is set up, run the master wrapper script:

```bash
python 00_wrapper.py
```

The wrapper will:
1. Scan `complete_data_116/` for all `.csv` files.
2. For each bacterium, execute sequentially:
   - `01_correlation.py`
   - `02_univariate_regression.py`
   - `03_multivariate_regression.py`
3. Generate a summary file `pipeline_summary.csv` in the main folder indicating success/failure for each script and bacterium.

---

## 📊 Output Structure

```
CORAL-python-scripts-results/
├── complete_data_116/               # Input data (your CSV files)
├── correlation_results/
│   └── <bacterium>/
│       ├── corrected_<bacterium>.csv          # Cleaned data (numeric, column names sanitized)
│       ├── ordinal_correction_log.csv         # Log of ordinal variable direction corrections
│       └── correlations_all.csv               # All pairwise Spearman correlations (ρ, p‑value, strength)
├── univariate_results/
│   └── <bacterium>/
│       └── univariate_outcome_results_<bacterium>.csv   # Univariate regression results for all outcomes
├── multivariate_results/
│   └── <bacterium>/
│       └── multivariate_full_puni0.2_<bacterium>.csv    # Multivariate regression results (final model)
├── 00_wrapper.py
├── 01_correlation.py
├── 02_univariate_regression.py
├── 03_multivariate_regression.py
└── pipeline_summary.csv              # Execution summary
```

### Output File Details

| File | Content |
|------|---------|
| `correlations_all.csv` | Columns: `variable1`, `variable2`, `correlation` (ρ), `p_value`, `strength` (Strong/Moderate/Weak), `n` (number of complete pairs). |
| `univariate_outcome_results_<bacterium>.csv` | Columns: `outcome`, `predictor`, `beta`, `p_value`, `ci_lower`, `ci_upper`, `regression_type` (linear/logistic/ordinal), `n` (sample size). |
| `multivariate_full_puni0.2_<bacterium>.csv` | Columns: `outcome`, `predictor`, `beta`, `p_value`, `ci_lower`, `ci_upper`, `regression_type`. Only predictors with multivariate *p* < 0.05 are included. |

---

## ❓ Troubleshooting Common Issues

| Problem | Possible Solution |
|---------|-------------------|
| `python: command not found` | Python is not installed or not in PATH. Reinstall and check "Add to PATH" (Windows) or ensure Python is in your shell's PATH (macOS/Linux). |
| `ModuleNotFoundError: No module named '...'` | Packages are missing. Run `pip install -r requirements.txt` again, or install dependencies manually. |
| `FileNotFoundError: complete_data_116/...` | Ensure your CSV files are inside `complete_data_116/` and that the folder name matches exactly. |
| Pipeline runs but no results | Check that your CSV data is strictly numeric (except `Patient.ID`) and that column names are clean (no spaces or special characters). |
| Slow performance | The correlation step is O(n²) in the number of variables. For large datasets, consider reducing the number of columns or using a more powerful machine. |
| `statsmodels` import errors on Apple Silicon | Use the force‑reinstall command provided above, or install from conda‑forge: `conda install statsmodels -c conda-forge`. |

---

## 📝 Customisation (Advanced)

All key parameters are hard‑coded in the scripts for simplicity, but can be easily modified:

| Parameter | Location | Default | Description |
|-----------|----------|---------|-------------|
| Univariate *p*-value threshold | `03_multivariate_regression.py`, line 17 | `0.2` | Predictors with univariate *p* < this value are considered for multivariate models. |
| Multivariate *p*-value threshold | `03_multivariate_regression.py`, line 18 | `0.05` | Only predictors with multivariate *p* < this value are kept in the final output. |
| Outcome list | `02_univariate_regression.py`, lines 23–26; `03_multivariate_regression.py`, lines 26–30 | 10 predefined outcomes | Add/remove outcomes by editing these lists. |
| Forbidden pairs | `03_multivariate_regression.py`, lines 25–42 | Mortality and stay outcome groups | Modify the groups to add/remove forbidden relationships. |
| Ordinal threshold | `02_univariate_regression.py`, line 34; `03_multivariate_regression.py`, line 55 | `≤10` unique values | Change this to adjust what is considered "ordinal" vs. "continuous". |

**To add more bacteria**: Simply drop new CSV files into `complete_data_116/`. The wrapper will automatically detect and process them.

---

## 📄 Citation

If you use this pipeline in your research, please cite this repository:

> Jain, A. *CORAL: Clinical Outcomes Regression and Analytical Learning Pipeline*. GitHub, 2026. [https://github.com/Anushka812/CORAL-python-scripts-results](https://github.com/Anushka812/CORAL-python-scripts-results)

---

## 📃 License

This pipeline is released under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## 🤝 Acknowledgements

Developed with clinical researchers in mind. Part of the documentation was prepared with the assistance of a large language model.

---

## 💬 Questions or Issues?

Please open an **Issue** on this GitHub repository – we will respond promptly.

---

**Happy analyzing!** 🧬📊
