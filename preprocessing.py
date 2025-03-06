import pandas as pd

# Step 1: Load the datasets
print("Loading Fake.csv...")
fake = pd.read_csv("Fake.csv")
print("Fake.csv Loaded!")

print("Loading True.csv...")
true = pd.read_csv("True.csv")
print("True.csv Loaded!")

# Step 2: Print the shape of both datasets (this will confirm they are loaded properly)
print("Fake dataset shape:", fake.shape)
print("True dataset shape:", true.shape)

# Step 3: Add a column to label fake (1) and real (0)
fake["label"] = 1
true["label"] = 0
print("Labels added to the datasets!")

# Step 4: Combine both datasets
df = pd.concat([fake, true], axis=0)
print("Combined dataset shape:", df.shape)

# Step 5: Shuffle the dataset
df = df.sample(frac=1).reset_index(drop=True)
print("Dataset shuffled!")

# Step 6: Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.2, random_state=42)
print("Train and Test split completed!")

# Step 7: Save train & test CSVs
train.to_csv("train.csv", index=False)
test.to_csv("test.csv", index=False)
print("Train and Test CSV files saved!")
