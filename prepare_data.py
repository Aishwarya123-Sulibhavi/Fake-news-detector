import pandas as pd

# Load the datasets
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

# Add a column to label fake (1) and real (0)
fake["label"] = 1
true["label"] = 0

# Combine both datasets
df = pd.concat([fake, true], axis=0)

# Shuffle the dataset
df = df.sample(frac=1).reset_index(drop=True)

# Split into train & test
from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.2, random_state=42)

# Save train & test CSVs
train.to_csv("train.csv", index=False)
test.to_csv("test.csv", index=False)
