import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
train_path = "insurance_preprocessing/insurance_train_preprocessed.csv"
test_path  = "insurance_preprocessing/insurance_test_preprocessed.csv"

train_df = pd.read_csv(train_path)
test_df  = pd.read_csv(test_path)

X_train = train_df.drop(columns=["target"])
y_train = train_df["target"]
X_test  = test_df.drop(columns=["target"])
y_test  = test_df["target"]

# Aktifkan autolog
mlflow.sklearn.autolog()

# Jalankan training (tanpa mlflow.start_run() — sudah dihandle oleh mlflow run)
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)

mlflow.sklearn.log_model(sk_model=model, artifact_path="model")

print(f"MSE : {mse:.4f}")
print(f"R2  : {r2:.4f}")
