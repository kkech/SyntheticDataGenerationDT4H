import pandas as pd
import torch
from sdv.metadata import SingleTableMetadata
from sdv.tabular import CTGAN
import os

def run_gpu_synthesis():
    # 1. Hardware Check
    print("--- 🖥️ HARDWARE AUDIT ---")
    if torch.cuda.is_available():
        device = torch.cuda.get_device_name(0)
        print(f"✅ GPU DETECTED: {device}")
        print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    else:
        print("⚠️ GPU NOT DETECTED: Falling back to CPU. (This will be much slower)")

    # 2. Load the clean data
    input_path = "/mnt/data/DT4Hnew/DT4H_Synthesis_Ready.parquet"
    output_csv = "/mnt/data/DT4Hnew/DT4H_SYNTHETIC_PATIENTS.csv"
    
    if not os.path.exists(input_path):
        print("❌ Error: Synthesis-ready file not found.")
        return

    print("\n--- 📂 LOADING CLEAN DATA ---")
    df = pd.read_parquet(input_path)

    # 3. Map the Data DNA (Metadata)
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=df)
    
    print(f"\n--- 🧬 DATA DNA MAPPED ---")
    print(f"Total Features: {len(df.columns)}")

    # 4. Initialize and Train CTGAN
    # CTGAN will automatically try to use CUDA if torch.cuda.is_available() is True
    print("\n--- 🤖 TRAINING AI (CTGAN) ON GPU ---")
    
    # Increase 'epochs' if you want higher data quality (default is usually 300)
    model = CTGAN(metadata, epochs=500, verbose=True) 
    model.fit(df)

    # 5. Generate Synthetic Patients
    print("\n--- 🧪 GENERATING 5,000 SYNTHETIC RECORDS ---")
    synthetic_data = model.sample(num_rows=5000)
    
    # 6. Save the final result
    synthetic_data.to_csv(output_csv, index=False)
    print(f"\n🎉 SUCCESS! Synthetic dataset saved to:\n{output_csv}")

if __name__ == "__main__":
    run_gpu_synthesis()
