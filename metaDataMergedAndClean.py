import polars as pl
import os

def generate_synthesis_metadata():
    file_path = "/mnt/data/DT4Hnew/DT4H_Cleaned_Data.parquet"
    output_path = "/mnt/data/DT4Hnew/DT4H_Synthesis_Ready.parquet"
    
    if not os.path.exists(file_path):
        print(f"❌ Error: Could not find {file_path}")
        return

    df = pl.read_parquet(file_path)
    total_rows = df.height
    print(f"--- 📊 ANALYZING: {total_rows} rows ---")
    
    cols_to_drop = []
    id_patterns = ['id_', 'contactid', 'metingid', 'prescriptionid']

    for col in df.columns:
        col_lower = col.lower()
        unique_count = df[col].n_unique()
        dtype = df[col].dtype
        
        # 1. ALWAYS drop pseudo_id
        if col_lower == "pseudo_id":
            cols_to_drop.append(col)
            continue

        # 2. GLOBAL HIGH-CARDINALITY CHECK (The "meettijd" fix)
        # If it's a string and has > 50 values, it's noise for the GAN.
        if dtype == pl.String and unique_count > 50:
            print(f"🗑️ Dropping '{col}': High cardinality ({unique_count} values).")
            cols_to_drop.append(col)
            continue

        # 3. Surgical ID drop
        if any(pat in col_lower for pat in id_patterns):
            cols_to_drop.append(col)
            continue

        # 4. Drop Constants
        if unique_count <= 1:
            cols_to_drop.append(col)
            continue

    df_synthesis = df.drop(cols_to_drop)
    df_synthesis.write_parquet(output_path)
    print(f"\n✅ SUCCESS: Reduced to {df_synthesis.width} clean features.")

if __name__ == "__main__":
    generate_synthesis_metadata()
