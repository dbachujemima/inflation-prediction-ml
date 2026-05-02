import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("../data/global_inflation_post_covid.csv")

# -----------------------------
# Inspect Columns (IMPORTANT)
# -----------------------------
print("Columns in dataset:")
print(df.columns)

# -----------------------------
# Data Cleaning
# -----------------------------
# Drop missing values
df = df.dropna()

# Convert categorical variables into numeric
df = pd.get_dummies(df, drop_first=True)

# -----------------------------
# Define Target Variable
# -----------------------------
# CHANGE THIS if your column name is different
target_column = "inflation_rate"

if target_column not in df.columns:
    raise ValueError(f"Column '{target_column}' not found. Check dataset column names.")

X = df.drop(target_column, axis=1)
y = df[target_column]

# -----------------------------
# Train/Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Model 1: Linear Regression
# -----------------------------
lr = LinearRegression()
lr.fit(X_train, y_train)
pred_lr = lr.predict(X_test)

print("\n--- Linear Regression ---")
print("MSE:", mean_squared_error(y_test, pred_lr))
print("R2 Score:", r2_score(y_test, pred_lr))

# -----------------------------
# Model 2: Random Forest
# -----------------------------
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)

print("\n--- Random Forest ---")
print("MSE:", mean_squared_error(y_test, pred_rf))
print("R2 Score:", r2_score(y_test, pred_rf))
