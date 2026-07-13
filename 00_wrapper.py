#!/usr/bin/env python3
"""
Run correlation, univariate, and multivariate analyses for every bacteria in complete_data_116/.
Each script must accept one command-line argument: the bacteria name (e.g., "pseudomonas").
"""

import os
import subprocess
import sys

# ========== CONFIGURATION ==========
DATA_FOLDER = "complete_data_116"        # folder containing all bacteria CSV files
SCRIPTS = [
    ("01_correlation.py", "correlation_results"),
    #("02_univariate_regression.py", "univariate_results"),
    #("03_multivariate_regression.py", "multivariate_results")
]
# ===================================

def get_bacteria_list():
    """Return list of bacteria names (without .csv extension)."""
    if not os.path.isdir(DATA_FOLDER):
        print(f"❌ Error: Data folder '{DATA_FOLDER}' not found.")
        sys.exit(1)
    csv_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('.csv')]
    if not csv_files:
        print(f"❌ Error: No CSV files found in '{DATA_FOLDER}'.")
        sys.exit(1)
    # Assume file names like "pseudomonas.csv", "ecoli.csv", etc.
    bacteria_list = [f.replace('.csv', '') for f in csv_files]
    return bacteria_list

def run_script(script_path, bacteria):
    """Run a single script for one bacteria."""
    print(f"\n▶ Running: {script_path} for {bacteria}")
    result = subprocess.run(
        [sys.executable, script_path, bacteria],
        capture_output=False,
        text=True
    )
    if result.returncode != 0:
        print(f"   ❌ {script_path} failed for {bacteria} (exit code {result.returncode})")
        return False
    print(f"   ✅ {script_path} completed for {bacteria}")
    return True

def main():
    bacteria_list = get_bacteria_list()
    print(f"🚀 Starting pipeline for {len(bacteria_list)} bacteria: {', '.join(bacteria_list)}")
    
    summary = []
    
    for bacteria in bacteria_list:
        print(f"\n{'='*70}")
        print(f"📊 PROCESSING BACTERIA: {bacteria}")
        print(f"{'='*70}")
        
        for script_path, output_folder in SCRIPTS:
            if not os.path.isfile(script_path):
                print(f"⚠️ Script not found: {script_path} - skipping.")
                summary.append((bacteria, script_path, False, "script missing"))
                continue
            
            success = run_script(script_path, bacteria)
            summary.append((bacteria, script_path, success, ""))
    
    # Final summary
    print("\n\n" + "="*70)
    print("PIPELINE COMPLETED - SUMMARY")
    print("="*70)
    for bact, script, ok, err in summary:
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"{status}  {bact:20} → {script}")
    
    # Save summary to CSV
    import pandas as pd
    df_summary = pd.DataFrame(summary, columns=["bacteria", "script", "success", "error"])
    df_summary.to_csv("pipeline_summary.csv", index=False)
    print("\n📄 Detailed summary saved to pipeline_summary.csv")

if __name__ == "__main__":
    main()