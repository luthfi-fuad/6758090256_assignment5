import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder

# Load the dataset
file_path = 'data/tourism_data.csv'  # Replace with your actual file path
df = pd.read_csv(file_path)

# Drop the rows where 'Occupation' is 'Free Lancer'
df = df[df['Occupation'] != 'Free Lancer']

# Fix typos in the 'Gender' column regardless of capitalization
df['Gender'] = df['Gender'].replace(
    to_replace=[r'(?i)fe male', r'(?i)female'], value='Female', regex=True)

# Standardize 'MaritalStatus' by replacing 'Unmarried' with 'Single' regardless of capitalization
df['MaritalStatus'] = df['MaritalStatus'].replace(
    to_replace=[r'(?i)Unmarried'], value='Single', regex=True)

# Now we will proceed to label encode the categorical columns
label_encoder = LabelEncoder()

# After transforming all categorical variables, save the transformed dataset
df.to_csv('data/tourism_data_cleaned.csv', index=False)

# Displaying the first few rows of the transformed dataset
print(df.head())
