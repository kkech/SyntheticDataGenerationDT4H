import pandas as pd
import torch
import os
from snsynth import Synthesizer

def run_dp_synthesis():
    # --- CONFIGURATION ---
    algorithm_name = "dpctgan"
    epsilon_budget = 15.0 
    
    input_path = "/mnt/data/DT4Hnew/DT4H_Synthesis_Ready.parquet"
    # Dynamically name the file based on algorithm and epsilon
    output_csv = f"/mnt/data/DT4Hnew/DT4H_{algorithm_name.upper()}_eps{epsilon_budget}.csv"
    
    if not torch.cuda.is_available():
        print("⚠️ GPU NOT DETECTED: DP-CTGAN requires a GPU for 98 features.")
        return
    else:
        print(f"✅ GPU DETECTED: {torch.cuda.get_device_name(0)}")

    print("\n--- 📂 LOADING & PREPPING DATA ---")
    df = pd.read_parquet(input_path)

    # Convert dates to strings to prevent DP-CTGAN from crashing
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].astype(str)

    # Automatically identify categorical columns (Strings)
    categorical_cols = [
        col for col in df.columns 
        if df[col].dtype == 'object' or str(df[col].dtype) in ['string', 'category']
    ]
    
    print(f"Detected {len(categorical_cols)} categorical features.")

    print("\n--- 🔒 INITIALIZING SYNTHESIZER ---")
    print(f"Algorithm: {algorithm_name.upper()}")
    print(f"Privacy Budget (ε): {epsilon_budget}")
    
    # Initialize the Synthesizer
    synth = Synthesizer.create(algorithm_name, epsilon=epsilon_budget, epochs=300)

    print("\n--- 🤖 TRAINING WITH DIFFERENTIAL PRIVACY ON GPU ---")
    print("Injecting mathematical noise... This will take a few minutes.")
    synth.fit(df, categorical_columns=categorical_cols)

    print("\n--- 🧪 GENERATING 5,000 DP-PROTECTED RECORDS ---")
    synthetic_data = synth.sample(5000)
    
    # Add a clean ID
    synthetic_data.insert(0, 'synthetic_id', [f"DT4H_DP_{i:04d}" for i in range(len(synthetic_data))])
    
    # Save with dynamic filename
    synthetic_data.to_csv(output_csv, index=False)
    print(f"\n🎉 SUCCESS! Dataset saved to:\n{output_csv}")

if __name__ == "__main__":
    run_dp_synthesis()
