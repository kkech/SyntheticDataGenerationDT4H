import pandas as pd
import os
import glob
from functools import reduce

def prepare_data_for_synthesis():
    folder_path = "/mnt/data/DT4Hnew/"
    
    # Grab all .parquet files in the folder
    parquet_files = glob.glob(os.path.join(folder_path, "*.parquet"))
    
    if not parquet_files:
        print(f"No .parquet files found in {folder_path}. Please check the path.")
        return

    dataframes = {}
    col_sets = []

    # 1. Read files, convert columns to lowercase, and print schemas
    print("--- 📂 READING FILES & FORMATTING COLUMNS ---")
    for file in parquet_files:
        file_name = os.path.basename(file)
        try:
            # Read the parquet file
            df = pd.read_parquet(file)
            
            # --- NEW STEP: Convert all column names to lowercase ---
            df.columns = df.columns.str.lower()
            
            dataframes[file_name] = df
            col_sets.append(set(df.columns))
            
            print(f"\n📄 File: {file_name}")
            print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
            print("Columns and Types:")
            print(df.dtypes.to_string())
            
        except Exception as e:
            print(f"Error reading {file_name}: {e}")

    # 2. Find the common column(s) for merging
    common_columns = list(set.intersection(*col_sets))
    
    print("\n--- 🔗 MERGING DATA ---")
    if not common_columns:
        print("WARNING: No common columns were found across ALL files.")
        print("You will need to manually specify the join key in the script.")
        # Manual override: Using lowercase fallback based on our new formatting
        common_columns = ['patientid'] 
    else:
        print(f"Common columns found for merging: {common_columns}")

    # 3. Merge all dataframes
    dfs_list = list(dataframes.values())
    try:
        merged_df = reduce(lambda left, right: pd.merge(left, right, on=common_columns, how='outer'), dfs_list)
        print(f"Merged Dataframe Shape: {merged_df.shape[0]} rows, {merged_df.shape[1]} columns")
    except Exception as e:
        print(f"Error during merging: {e}")
        return

    # 4. Data Cleansing
    print("\n--- 🧹 DATA CLEANSING ---")
    
    # a) Drop columns with > 80% nulls
    threshold_pct = 0.80
    min_non_nulls = int((1 - threshold_pct) * len(merged_df))
    
    print("Dropping columns with more than 80% missing values...")
    cleaned_df = merged_df.dropna(axis=1, thresh=min_non_nulls)
    print(f"Shape after dropping sparse columns: {cleaned_df.shape[0]} rows, {cleaned_df.shape[1]} columns")

    # b) Drop rows with ANY nulls
    print("Dropping rows that contain any missing values...")
    final_df = cleaned_df.dropna(axis=0, how='any')
    print(f"Final Dataframe Shape: {final_df.shape[0]} rows, {final_df.shape[1]} columns")

    # 5. Save the final file
    output_filename = "DT4H_ReadyForSynth.parquet"
    output_path = os.path.join(folder_path, output_filename)
    
    print("\n--- 💾 SAVING ---")
    try:
        final_df.to_parquet(output_path, index=False)
        print(f"Successfully saved cleaned data to:\n{output_path}")
    except Exception as e:
        print(f"Error saving file: {e}")

if __name__ == "__main__":
    prepare_data_for_synthesis()
