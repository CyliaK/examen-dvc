import pandas as pd
from sklearn.model_selection import train_test_split

# Chemins des fichiers
RAW_DATA_PATH = "data/raw_data/raw.csv"
X_train_path = "data/processed_data/X_train.csv"
X_test_path  = "data/processed_data/X_test.csv"
y_train_path = "data/processed_data/y_train.csv"
y_test_path  = "data/processed_data/y_test.csv"

# Charger le dataset
df = pd.read_csv(RAW_DATA_PATH)

# Séparer X et y
X = df.drop(columns=["silica_concentrate"])
y = df["silica_concentrate"]

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Sauvegarder les fichiers CSV
X_train.to_csv(X_train_path, index=False)
X_test.to_csv(X_test_path, index=False)
y_train.to_csv(y_train_path, index=False)
y_test.to_csv(y_test_path, index=False)

print("Split terminé, fichiers créés dans data/processed_data")