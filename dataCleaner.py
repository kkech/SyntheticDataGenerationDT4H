import polars as pl
import os
import glob

def prepare_data_pairwise_polars():
    folder_path = "/mnt/data/DT4Hnew/"
    parquet_files = glob.glob(os.path.join(folder_path, "*.parquet"))
    
    if not parquet_files:
        print(f"No .parquet files found in {folder_path}.")
        return

    target_id = 'pseudo_id'
    lazy_dfs = []

    print("--- 📂 STEP 1: SCANNING FILES & FORMATTING ---")
    for file in parquet_files:
        file_name = os.path.basename(file)
        
        lf = pl.scan_parquet(file)
        
        # Convert columns to lowercase safely
        lowercase_mapping = {col: col.lower() for col in lf.collect_schema().names()}
        lf = lf.rename(lowercase_mapping)
        
        if target_id in lf.collect_schema().names():
            lazy_dfs.append(lf)
        else:
            print(f"  -> Skipping {file_name} (No '{target_id}' found)")

    if not lazy_dfs:
        print(f"CRITICAL: '{target_id}' not found in any files. Aborting.")
        return

    print("\n--- 🔗 STEP 2: BUILDING PAIRWISE MERGE GRAPH ---")
    iteration = 1
    
    # Loop until all LazyFrames are merged into a single one
    while len(lazy_dfs) > 1:
        print(f"Merge Pass {iteration} | Dataframes to merge: {len(lazy_dfs)}")
        next_layer = []
        
        # Process in pairs (0 & 1, 2 & 3, etc.)
        for i in range(0, len(lazy_dfs), 2):
            # If we have a pair to merge
            if i + 1 < len(lazy_dfs):
                lf1 = lazy_dfs[i]
                lf2 = lazy_dfs[i+1]
                
                # Find common columns for this specific pair
                cols1 = set(lf1.collect_schema().names())
                cols2 = set(lf2.collect_schema().names())
                common_cols = list(cols1 & cols2)
                
                # Full join keeps all patients. Coalesce ensures the pseudo_ids 
                # merge into one column instead of duplicating.
                merged_lf = lf1.join(lf2, on=common_cols, how="full", coalesce=True)
                next_layer.append(merged_lf)
            else:
                # Odd one out: just pass it to the next round
                next_layer.append(lazy_dfs[i])
                
        # Update our list of frames for the next iteration
        lazy_dfs = next_layer
        iteration += 1

    # The final remaining LazyFrame contains the entire graph
    master_lf = lazy_dfs[0]

    print("\n--- 🚀 STEP 3: EXECUTING MERGE (COLLECTING DATA) ---")
    # streaming=True protects the RAM during heavy full joins
    merged_df = master_lf.collect(streaming=True)
    print(f"Merged Dataframe Shape: {merged_df.height} rows, {merged_df.width} columns")

    print("\n--- 🧹 STEP 4: DATA CLEANSING ---")
    # a) Drop columns with > 80% nulls
    print("Dropping columns with more than 80% missing values...")
    null_counts = merged_df.null_count()
    num_rows = merged_df.height
    
    cols_to_keep = [
        col for col in merged_df.columns 
        if null_counts[col].item() / num_rows <= 0.80
    ]
    
    cleaned_df = merged_df.select(cols_to_keep)
    print(f"Shape after dropping sparse columns: {cleaned_df.height} rows, {cleaned_df.width} columns")

    # b) Drop rows with ANY nulls
    print("Dropping rows that contain any missing values...")
    final_df = cleaned_df.drop_nulls()
    print(f"Final Dataframe Shape: {final_df.height} rows, {final_df.width} columns")

    # --- SAVE ---
    output_filename = "DT4H_Pairwise_Polars.parquet"
    output_path = os.path.join(folder_path, output_filename)
    
    print("\n--- 💾 SAVING ---")
    final_df.write_parquet(output_path)
    print(f"Successfully saved to:\n{output_path}")

if __name__ == "__main__":
    prepare_data_pairwise_polars()
