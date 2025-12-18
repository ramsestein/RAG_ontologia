import pandas as pd

# 1. Load your raw Excel file
# Note: Ensure the filename matches what is on your disk
input_file = "../data/taxonomia.xslx" 
df = pd.read_csv(input_file)

# 2. Function to fix "Excel-damaged" codes
def clean_terminology_code(value):
    if pd.isna(value) or value == "":
        return None
    try:
        # Convert scientific notation (1.62E+16) -> Float -> Int -> String
        # This recovers the full number: "16218291000119100"
        return str(int(float(value)))
    except ValueError:
        # If it's already a string or text code, just return it stripped
        return str(value).strip()

# 3. Apply the cleaning
df['clean_code'] = df['terminology_code'].apply(clean_terminology_code)

# 4. Filter: Keep only what we need
# We select the preferred name, synonyms, and the clean code
# You might want to keep 'nombre_local_hallazgo' as your main label
clean_df = df[['nombre_local_hallazgo', 'clean_code', 'preferido', 'sinonimo', 'sinonimo_1']].copy()

# 5. Save as the definitive 'taxonomia.csv'
clean_df.to_csv("taxonomia_cleaned.csv", index=False)

# --- Verification ---
print("✅ File saved as 'taxonomia_cleaned.csv'")
print("Preview of fixed codes:")
print(clean_df[['nombre_local_hallazgo', 'clean_code']].head())