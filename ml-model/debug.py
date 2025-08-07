import pandas as pd

# Load and check the data
df = pd.read_csv('../data/credit_data.csv')

print("Dataset shape:", df.shape)
print("\nTarget distribution:")
print(df['credit_worthy'].value_counts())
print("\nTarget value counts (percentages):")
print(df['credit_worthy'].value_counts(normalize=True))

print("\nCredit score range:")
print(f"Min: {df['credit_score'].min()}, Max: {df['credit_score'].max()}")

print("\nFirst few rows:")
print(df.head())