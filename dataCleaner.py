import pandas as pd
import os
import glob
from functools import reduce

def prepare_data_for_synthesis():
    # 1. Define the directory based on your screenshot
    folder_path = r"Z:\DT4Hnew"
    
    # Grab all .parquet files in the folder
    parquet_files = glob.glob(os.path.join(folder_path, "*.parquet"))
    
    if not parquet_files:
        print(f"No .parquet files found in {folder_path}. Please check the path.")
        return

    dataframes = {}
    col_sets = []

    # 2. Read files and print columns & data types to the terminal
    print("--- 📂 READING FILES & PRINTING SCHEMAS ---")
    for file in parquet_files:
        file_name = os.path.basename(file)
        try:
            # Read the parquet file
            df = pd.read_parquet(file)
            dataframes[file_name] = df
            col_sets.append(set(df.columns))
            
            print(f"\n📄 File: {file_name}")
            print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
            print("Columns and Types:")
            print(df.dtypes.to_string()) # to_string() ensures it prints all columns without truncating
            
        except Exception as e:
            print(f"Error reading {file_name}: {e}")

    # 3. Find the common column(s) for merging
    # This looks for columns that exist in EVERY loaded dataframe (e.g., 'PatientID')
    common_columns = list(set.intersection(*col_sets))
    
    print("\n--- 🔗 MERGING DATA ---")
    if not common_columns:
        print("WARNING: No common columns were found across ALL files.")
        print("You will need to manually specify the join key in the script.")
        # Manual override: replace 'PatientID' with the actual ID column name from your terminal output
        common_columns = ['PatientID'] 
    else:
        print(f"Common columns found for merging: {common_columns}")

    # Merge all dataframes. Using an 'outer' join to initially keep all patient data
    # before we start the cleansing process.
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
    # pandas dropna 'thresh' requires the *minimum number of non-NA values* to keep the column.
    # So if we want to drop >80% nulls, we require at least 20% to be non-null.
    min_non_nulls = int((1 - threshold_pct) * len(merged_df))
    
    print(f"Dropping columns with more than 80% missing values...")
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
