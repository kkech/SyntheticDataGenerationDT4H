import pandas as pd

def verify_synthetic_data():
    path = "/mnt/data/DT4Hnew/DT4H_SYNTHETIC_PATIENTS.csv"
    
    # Load the synthetic data
    df = pd.read_csv(path)
    
    print(f"--- 📊 SYNTHETIC DATA AUDIT ---")
    print(f"Shape: {df.shape[0]} Rows, {df.shape[1]} Columns")
    
    print("\n--- 🧐 FIRST 5 RECORDS ---")
    print(df.head())
    
    print("\n--- 📉 DATA COMPLETENESS ---")
    # Check if the AI generated too many nulls or perfect data
    null_counts = df.isnull().sum().sum()
    total_cells = df.size
    print(f"Total Null Cells: {null_counts} ({null_counts/total_cells:.2%} of total data)")
    
    # Check unique values for a clinical column to see if it looks diverse
    # (I'll pick a column name from your previous screenshots)
    col = 'ziekenhuislocatie_dt4h_bloodpressure'
    if col in df.columns:
        print(f"\n--- 🏥 COLUMN CHECK: {col} ---")
        print(df[col].value_counts())

if __name__ == "__main__":
    verify_synthetic_data()
