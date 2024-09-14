import pandas as pd
from sklearn.model_selection import train_test_split

# Load your data
df = pd.read_csv('dataset/train.csv')

# Split the data into train and test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Save the split datasets to new CSV files
train_df.to_csv('train_split.csv', index=False)
test_df.to_csv('test_split.csv', index=False)
