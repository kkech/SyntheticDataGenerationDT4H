import pandas as pd
import torch
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer
import os

def run_gpu_synthesis():
    input_path = "/mnt/data/DT4Hnew/DT4H_Synthesis_Ready.parquet"
    output_csv = "/mnt/data/DT4Hnew/DT4H_SYNTHETIC_PATIENTS.csv"
    
    if not torch.cuda.is_available():
        print("⚠️ GPU NOT DETECTED: Falling back to CPU.")
    else:
        print(f"✅ GPU DETECTED: {torch.cuda.get_device_name(0)}")

    print("--- 📂 LOADING DATA ---")
    df = pd.read_parquet(input_path)

    # 1. Map Metadata
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=df)
    
    # 2. Initialize CTGANSynthesizer
    # In newer SDV, we use CTGANSynthesizer
    model = CTGANSynthesizer(metadata, epochs=500)

    # 3. Train
    print("\n--- 🤖 TRAINING ON GPU ---")
    model.fit(df)

    # 4. Generate
    print("\n--- 🧪 GENERATING 5,000 RECORDS ---")
    synthetic_data = model.sample(num_rows=5000)
    
    synthetic_data.to_csv(output_csv, index=False)
    print(f"\n🎉 SUCCESS! Saved to: {output_csv}")

if __name__ == "__main__":
    run_gpu_synthesis()
