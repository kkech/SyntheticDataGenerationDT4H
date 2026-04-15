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
        
        rename_map = {col: (col.lower() if col.lower() == target_id else f"{col.lower()}_{file_base}") 
                      for col in lf.collect_schema().names()}
        
        lf = lf.rename(rename_map)
        
        if target_id in lf.collect_schema().names():
            # Standardize ID and keep one row per patient
            lf = lf.with_columns(pl.col(target_id).cast(pl.String))
            lf = lf.unique(subset=[target_id], keep="first")
            lazy_dfs.append(lf)
        else:
            print(f"  -> Skipping {file_name} (No '{target_id}' found)")

    print("\n--- 🔗 STEP 2: PAIRWISE MERGE ---")
    while len(lazy_dfs) > 1:
        next_layer = []
        for i in range(0, len(lazy_dfs), 2):
            if i + 1 < len(lazy_dfs):
                next_layer.append(lazy_dfs[i].join(lazy_dfs[i+1], on=[target_id], how="full", coalesce=True))
            else:
                next_layer.append(lazy_dfs[i])
        lazy_dfs = next_layer

    print("\n--- 🚀 STEP 3: EXECUTING MERGE ---")
    merged_df = lazy_dfs[0].collect(engine="streaming")
    initial_rows = merged_df.height
    print(f"Total unique patients found: {initial_rows}")

    print("\n--- 📊 STEP 4: DATA STATISTICS & CLEANSING ---")
    
    # Calculate null percentage per column
    null_pcts = (merged_df.null_count() / initial_rows).to_dicts()[0]
    
    # Drop columns with > 80% nulls
    cols_to_keep = [col for col, pct in null_pcts.items() if pct <= 0.80]
    cleaned_df = merged_df.select(cols_to_keep)
    
    dropped_cols = len(merged_df.columns) - len(cols_to_keep)
    print(f"Dropped {dropped_cols} columns for being > 80% empty.")
    print(f"Remaining columns: {len(cols_to_keep)}")

    # Calculate row-wise statistics
    # Instead of drop_nulls (which is too strict), let's see how much data each row has
    # We define "completeness" as what % of medical columns are filled for that patient
    medical_cols = [c for c in cleaned_df.columns if c != target_id]
    
    # Filter rows: Keep patients who have at least 1 piece of medical data 
    # (prevents dropping everyone because of one missing lab)
    final_df = cleaned_df.filter(
        pl.any_horizontal(pl.col(medical_cols).is_not_null())
    )

    print(f"Rows before row-cleansing: {initial_rows}")
    print(f"Rows after removing totally empty patients: {final_df.height}")
    
    if final_df.height == 0:
        print("⚠️ WARNING: 0 rows remain. The datasets might not overlap at all on pseudo_id.")
    else:
        # --- SAVE ---
        output_path = os.path.join(folder_path, "DT4H_Cleaned_Data.parquet")
        final_df.write_parquet(output_path)
        print(f"\n✅ SUCCESS: Saved {final_df.height} rows to {output_path}")

if __name__ == "__main__":
    prepare_data_pairwise_polars()
