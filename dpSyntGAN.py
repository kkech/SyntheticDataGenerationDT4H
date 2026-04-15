import pandas as pd
import torch
import os
from snsynth import Synthesizer

def run_dp_synthesis():
    # --- CONFIGURATION ---
    algorithm_name = "dpctgan"
    epsilon_budget = 15.0 
    
    input_path = "/mnt/data/DT4Hnew/DT4H_Synthesis_Ready.parquet"
    output_csv = f"/mnt/data/DT4Hnew/DT4H_{algorithm_name.upper()}_eps{epsilon_budget}.csv"
    
    if not torch.cuda.is_available():
        print("⚠️ GPU NOT DETECTED: DP-CTGAN requires a GPU.")
        return
    else:
        print(f"✅ GPU DETECTED: {torch.cuda.get_device_name(0)}")

    print("\n--- 📂 LOADING & PREPPING DATA ---")
    df = pd.read_parquet(input_path)

    # 1. Convert dates to strings
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].astype(str)

    # 2. Explicitly define Continuous (Numerical) vs Categorical columns
    # We cast numericals to float to prevent Pandas integer-null crashes
    continuous_cols = []
    categorical_cols = []

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype(float)
            continuous_cols.append(col)
        else:
            categorical_cols.append(col)

    print(f"Mapped {len(categorical_cols)} Categorical and {len(continuous_cols)} Continuous features.")

    print("\n--- 🔒 INITIALIZING SYNTHESIZER ---")
    print(f"Algorithm: {algorithm_name.upper()}")
    print(f"Privacy Budget (ε): {epsilon_budget}")
    
    synth = Synthesizer.create(algorithm_name, epsilon=epsilon_budget, epochs=300)

    print("\n--- 🤖 TRAINING WITH DIFFERENTIAL PRIVACY ON GPU ---")
    print("Injecting mathematical noise... This will take a few minutes.")
    
    # FIX: Explicitly pass BOTH column types to the synthesizer
    synth.fit(
        df, 
        categorical_columns=categorical_cols, 
        continuous_columns=continuous_cols
    )

    print("\n--- 🧪 GENERATING 5,000 DP-PROTECTED RECORDS ---")
    synthetic_data = synth.sample(5000)
    
    synthetic_data.insert(0, 'synthetic_id', [f"DT4H_DP_{i:04d}" for i in range(len(synthetic_data))])
    
    synthetic_data.to_csv(output_csv, index=False)
    print(f"\n🎉 SUCCESS! Dataset saved to:\n{output_csv}")

if __name__ == "__main__":
    run_dp_synthesis()
