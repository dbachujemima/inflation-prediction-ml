import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("../data/global_inflation_post_covid.csv")

# -----------------------------
# Basic Info
# -----------------------------
print("\n--- First 5 Rows ---")
print(df.head())

print("\n--- Dataset Info ---")
print(df.info())

print("\n--- Summary Statistics ---")
print(df.describe())

print("\n--- Columns ---")
print(df.columns)

# -----------------------------
# Data Cleaning
# -----------------------------
# Drop missing values
df = df.dropna()

# -----------------------------
# Visualization 1: Inflation Distribution
# -----------------------------
if "inflation_rate" in df.columns:
    plt.figure()
    sns.histplot(df["inflation_rate"], kde=True)
    plt.title("Distribution of Inflation Rate")
    plt.xlabel("Inflation Rate")
    plt.ylabel("Frequency")
    plt.show()

# -----------------------------
# Visualization 2: Inflation Over Time (if available)
# -----------------------------
if "year" in df.columns and "inflation_rate" in df.columns:
    plt.figure()
    sns.lineplot(x="year", y="inflation_rate", data=df)
    plt.title("Inflation Trend Over Time")
    plt.xlabel("Year")
    plt.ylabel("Inflation Rate")
    plt.show()

# -----------------------------
# Visualization 3: Correlation Heatmap
# -----------------------------
# Convert categorical to numeric for correlation
df_encoded = pd.get_dummies(df, drop_first=True)

plt.figure(figsize=(10, 8))
sns.heatmap(df_encoded.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()
