#!/usr/bin/env python3
"""
Execute the complete statistical analysis pipeline for each bacterial dataset contained within the input directory.
For every bacterial species, the pipeline sequentially performs:
    1. Pairwise correlation analysis
    2. Univariate regression analysis
    3. Multivariate regression analysis
Each analytical script receives the bacterial species name as a command-line argument, enabling the same workflow to be applied consistently across all organism-specific datasets.
"""
import os
import subprocess
import sys

"""
Define the input data directory and the analytical workflow. The scripts are executed sequentially for each bacterial dataset, ensuring a consistent and reproducible analysis pipeline.
"""

DATA_FOLDER = "complete_data_116"        # Directory containing one CSV file per bacterial species.
SCRIPTS = [
    ("01_correlation.py", "correlation_results"),
    ("02_univariate_regression.py", "univariate_results"),
    ("03_multivariate_regression.py", "multivariate_results")
]
def get_bacteria_list():
    # Identify all bacterial datasets available for analysis.
    if not os.path.isdir(DATA_FOLDER):
        print(f" Error: Data folder '{DATA_FOLDER}' not found.")
        sys.exit(1)
    csv_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('.csv')]
    if not csv_files:
        print(f" Error: No CSV files found in '{DATA_FOLDER}'.")
        sys.exit(1)
    # Extract bacterial species names from the input filenames.
    bacteria_list = [f.replace('.csv', '') for f in csv_files]
    return bacteria_list

def run_script(script_path, bacteria):
    """
    Execute a single analytical script for one bacterial dataset. The wrapper monitors successful
    completion of each stage so that the overall pipeline can be summarised upon completion.
    """
    print(f"\nRunning: {script_path} for {bacteria}")
    result = subprocess.run(
        [sys.executable, script_path, bacteria],
        capture_output=False,
        text=True
    )
    if result.returncode != 0:
        print(f" {script_path} failed for {bacteria} (exit code {result.returncode})")
        return False
    print(f" {script_path} completed for {bacteria}")
    return True

def main():
    # Identify all bacterial datasets available for analysis.
    bacteria_list = get_bacteria_list()
    print(f" Starting pipeline for {len(bacteria_list)} bacteria: {', '.join(bacteria_list)}")
    # Record execution status for each analytical stage.
    summary = []

    # Execute the complete statistical workflow for each bacterial dataset.
    for bacteria in bacteria_list:
        print(f"\n{'='*70}")
        print(f" PROCESSING BACTERIA: {bacteria}")
        print(f"{'='*70}")
        # Perform correlation, univariate regression and multivariate regression analyses sequentially.
        for script_path, output_folder in SCRIPTS:
            if not os.path.isfile(script_path):
                print(f" Script not found: {script_path} - skipping.")
                summary.append((bacteria, script_path, False, "script missing"))
                continue
            success = run_script(script_path, bacteria)
            summary.append((bacteria, script_path, success, ""))

    # Summarise execution of the complete analysis pipeline.
    print("\n\n" + "="*70)
    print("PIPELINE COMPLETED - SUMMARY")
    print("="*70)
    for bact, script, ok, err in summary:
        status = " PASS" if ok else " FAIL"
        print(f"{status}  {bact:20} → {script}")

    # Export the execution summary to facilitate verification of successful completion for each
    # bacterial dataset.
    import pandas as pd
    df_summary = pd.DataFrame(
        summary,
        columns=["bacteria", "script", "success", "error"]
    )
    df_summary.to_csv("pipeline_summary.csv", index=False)
    print("\n Detailed summary saved to pipeline_summary.csv")

if __name__ == "__main__":
    main()
