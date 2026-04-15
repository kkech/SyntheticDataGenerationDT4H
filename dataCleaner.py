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

    print("--- 📂 STEP 1: FORMATTING & DEDUPLICATING ---")
    for file in parquet_files:
        file_name = os.path.basename(file)
        file_base = os.path.splitext(file_name)[0].lower() 
        
        lf = pl.scan_parquet(file)
        
        rename_map = {}
        for col in lf.collect_schema().names():
            col_lower = col.lower()
            if col_lower == target_id:
                rename_map[col] = target_id
            else:
                rename_map[col] = f"{col_lower}_{file_base}"
        
        lf = lf.rename(rename_map)
        
        if target_id in lf.collect_schema().names():
            lf = lf.with_columns(pl.col(target_id).cast(pl.String))
            
            # --- THE EXPLOSION FIX ---
            # Keep exactly ONE row per patient per file to stop Cartesian explosions.
            lf = lf.unique(subset=[target_id], keep="first")
            
            lazy_dfs.append(lf)
        else:
            print(f"  -> Skipping {file_name} (No '{target_id}' found)")

    if not lazy_dfs:
        print(f"CRITICAL: '{target_id}' not found in any files. Aborting.")
        return

    print("\n--- 🔗 STEP 2: BUILDING PAIRWISE MERGE GRAPH ---")
    iteration = 1
    
    while len(lazy_dfs) > 1:
        print(f"Merge Pass {iteration} | Dataframes to merge: {len(lazy_dfs)}")
        next_layer = []
        
        for i in range(0, len(lazy_dfs), 2):
            if i + 1 < len(lazy_dfs):
                lf1 = lazy_dfs[i]
                lf2 = lazy_dfs[i+1]
                
                merged_lf = lf1.join(
                    lf2, 
                    on=[target_id], 
                    how="full", 
                    coalesce=True
                )
                next_layer.append(merged_lf)
            else:
                next_layer.append(lazy_dfs[i])
                
        lazy_dfs = next_layer
        iteration += 1

    master_lf = lazy_dfs[0]

    print("\n--- 🚀 STEP 3: EXECUTING MERGE (COLLECTING DATA) ---")
    # --- THE ENGINE FIX ---
    # Updated to the modern Polars syntax to process the merge safely out-of-core
    merged_df = master_lf.collect(engine="streaming")
    print(f"Merged Dataframe Shape: {merged_df.height} rows, {merged_df.width} columns")

    print("\n--- 🧹 STEP 4: DATA CLEANSING ---")
    print("Dropping columns with more than 80% missing values...")
    null_counts = merged_df.null_count()
    num_rows = merged_df.height
    
    cols_to_keep = [
        col for col in merged_df.columns 
        if null_counts[col].item() / num_rows <= 0.80
    ]
    
    cleaned_df = merged_df.select(cols_to_keep)
    print(f"Shape after dropping sparse columns: {cleaned_df.height} rows, {cleaned_df.width} columns")

    print("Dropping rows that contain any missing values...")
    final_df = cleaned_df.drop_nulls()
    print(f"Final Dataframe Shape: {final_df.height} rows, {final_df.width} columns")

    output_filename = "DT4H_Pairwise_Polars.parquet"
    output_path = os.path.join(folder_path, output_filename)
    
    print("\n--- 💾 SAVING ---")
    final_df.write_parquet(output_path)
    print(f"Successfully saved to:\n{output_path}")

if __name__ == "__main__":
    prepare_data_pairwise_polars()
