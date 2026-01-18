import pandas as pd
from sklearn.preprocessing import StandardScaler

# Chemins des fichiers
PROCESSED_PATH = "data/processed_data"
X_train_path = f"{PROCESSED_PATH}/X_train.csv"
X_test_path  = f"{PROCESSED_PATH}/X_test.csv"
X_train_scaled_path = f"{PROCESSED_PATH}/X_train_scaled.csv"
X_test_scaled_path  = f"{PROCESSED_PATH}/X_test_scaled.csv"

# Charger les données
X_train = pd.read_csv(X_train_path)
X_test  = pd.read_csv(X_test_path)

# Garder uniquement les colonnes numériques
numeric_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
X_train_num = X_train[numeric_cols]
X_test_num  = X_test[numeric_cols]

# Normalisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_num)
X_test_scaled  = scaler.transform(X_test_num)

# Convertir en DataFrame et sauvegarder
X_train_scaled = pd.DataFrame(X_train_scaled, columns=numeric_cols)
X_test_scaled  = pd.DataFrame(X_test_scaled, columns=numeric_cols)

X_train_scaled.to_csv(X_train_scaled_path, index=False)
X_test_scaled.to_csv(X_test_scaled_path, index=False)

print("Normalisation terminée, fichiers X_train_scaled.csv et X_test_scaled.csv créés")