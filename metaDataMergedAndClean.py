import polars as pl
import os

def generate_synthesis_metadata():
    # Input: The merged file
    file_path = "/mnt/data/DT4Hnew/DT4H_Cleaned_Data.parquet"
    output_path = "/mnt/data/DT4Hnew/DT4H_Synthesis_Ready.parquet"
    
    if not os.path.exists(file_path):
        print(f"❌ Error: Could not find {file_path}")
        return

    # Load the data
    df = pl.read_parquet(file_path)
    total_rows = df.height
    
    print(f"--- 📊 ANALYZING STRUCTURE: {total_rows} rows ---")
    
    cols_to_drop = []
    metadata = []

    # Keywords for surgical ID removal
    id_patterns = ['id_', 'contactid', 'metingid', 'prescriptionid']

    # 1. Audit each column
    for col in df.columns:
        col_lower = col.lower()
        unique_count = df[col].n_unique()
        dtype = df[col].dtype
        null_count = df[col].null_count()
        
        # --- A) Drop Primary ID ---
        if col_lower == "pseudo_id":
            cols_to_drop.append(col)
            print(f"🗑️ Dropping '{col}': Primary Identifier")
            continue

        # --- B) CTGAN SAFETY: Drop High-Cardinality Categoricals ---
        # If it's a String and has > 50 unique values (like meettijd or timestamps),
        # CTGAN will struggle. We drop these to ensure the model can train.
        if dtype == pl.String and unique_count > 50:
            cols_to_drop.append(col)
            print(f"⚠️ Dropping '{col}': Too many categories ({unique_count}). This prevents CTGAN crashes.")
            continue

        # --- C) Surgical Drop: Transaction/Contact IDs ---
        if any(pat in col_lower for pat in id_patterns):
            if unique_count > (total_rows * 0.10): 
                cols_to_drop.append(col)
                print(f"✂️ Surgical Drop '{col}': High-cardinality Transaction ID")
                continue

        # --- D) Drop Constant Columns ---
        if unique_count <= 1:
            cols_to_drop.append(col)
            print(f"🗑️ Dropping '{col}': Constant Value")
            continue

        # If it survives, add to report
        metadata.append({
            "name": col,
            "type": str(dtype),
            "uniques": unique_count,
            "nulls": null_count
        })

    # Execute all drops
    df_synthesis = df.drop(cols_to_drop)

    # 2. Print the Final Metadata Table
    print("\n" + "="*90)
    print(f"{'COLUMN NAME':<50} | {'TYPE':<15} | {'UNIQUE':<10} | {'NULLS'}")
    print("-" * 90)
    
    for item in metadata:
        print(f"{item['name']:<50} | {item['type']:<15} | {item['uniques']:<10} | {item['nulls']}")
    print("="*90)

    # 3. Summary
    cat_cols = [c for c in df_synthesis.columns if df_synthesis[c].dtype == pl.String]
    num_cols = [c for c in df_synthesis.columns if df_synthesis[c].dtype in [pl.Int32, pl.Int64, pl.Float32, pl.Float64]]
    date_cols = [c for c in df_synthesis.columns if df_synthesis[c].dtype in [pl.Date, pl.Datetime]]
    
    print(f"\n✅ DATASET READY FOR SYNTHESIS")
    print(f"Final Features:      {df_synthesis.width}")
    print(f"Categorical Strings: {len(cat_cols)} (Max 50 unique values per column)")
    print(f"Numerical Values:    {len(num_cols)}")
    print(f"Date/Time Columns:   {len(date_cols)}")
    
    # 4. Save
    df_synthesis.write_parquet(output_path)
    print(f"\n💾 Saved synthesis-ready file to: {output_path}")

if __name__ == "__main__":
    generate_synthesis_metadata()
