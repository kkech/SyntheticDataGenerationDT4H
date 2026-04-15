import polars as pl
import os

def generate_synthesis_metadata():
    # The path to your cleaned dataset from the previous step
    file_path = "/mnt/data/DT4Hnew/DT4H_Cleaned_Data.parquet"
    
    if not os.path.exists(file_path):
        print(f"❌ Error: Could not find {file_path}")
        return

    # Load the data
    df = pl.read_parquet(file_path)
    total_rows = df.height
    
    print(f"--- 📊 ANALYZING STRUCTURE: {total_rows} rows ---")
    
    cols_to_drop = []
    metadata = []

    # 1. Audit each column to see if it's "Generatable"
    for col in df.columns:
        unique_count = df[col].n_unique()
        dtype = df[col].dtype
        null_count = df[col].null_count()
        
        # LOGIC: Identifying columns that break synthesis
        
        # A) Drop Unique Identifiers
        # Synthetic generators can't "learn" an ID. We drop pseudo_id here. 
        # You should re-generate IDs later in your synthetic output.
        if col == "pseudo_id":
            reason = "Unique ID (Noise)"
            cols_to_drop.append(col)
            print(f"🗑️ Dropping '{col}': {reason}")
            continue

        # B) Drop Constant Columns
        # If every patient has the same value, there is no variance for the AI to learn.
        if unique_count == 1:
            reason = "Constant Value"
            cols_to_drop.append(col)
            print(f"🗑️ Dropping '{col}': {reason}")
            continue

        # If it survives, add to metadata report
        metadata.append({
            "name": col,
            "type": str(dtype),
            "uniques": unique_count,
            "nulls": null_count
        })

    # Execute drops
    df_synthesis = df.drop(cols_to_drop)

    # 2. Print the Final Metadata Table
    print("\n" + "="*85)
    print(f"{'COLUMN NAME':<45} | {'TYPE':<15} | {'UNIQUE':<10} | {'NULLS'}")
    print("-" * 85)
    
    for item in metadata:
        print(f"{item['name']:<45} | {item['type']:<15} | {item['uniques']:<10} | {item['nulls']}")
    print("="*85)

    # 3. Summary for your Synthetic Generator Configuration
    cat_cols = [c for c in df_synthesis.columns if df_synthesis[c].dtype == pl.String]
    num_cols = [c for c in df_synthesis.columns if df_synthesis[c].dtype in [pl.Int32, pl.Int64, pl.Float32, pl.Float64]]
    
    print(f"\n✅ DATASET READY FOR SYNTHESIS ENGINE")
    print(f"Total Features to Model: {df_synthesis.width}")
    print(f"Categorical Features:   {len(cat_cols)} (Will need Categorical Transformers)")
    print(f"Numerical Features:     {len(num_cols)} (Will need Scalar/Gaussian Transformers)")
    
    # Optional: Save a final version that is "pure" for the AI
    output_path = "/mnt/data/DT4Hnew/DT4H_Synthesis_Ready.parquet"
    df_synthesis.write_parquet(output_path)
    print(f"\n💾 Saved synthesis-ready file to: {output_path}")

if __name__ == "__main__":
    generate_synthesis_metadata()
