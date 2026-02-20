import pandas as pd
import numpy as np

def clean_data(df):
    """Deep feature engineering for clean (unencoded) dataset."""
    # Note: Strings are already standardized (Gender/MaritalStatus) in tourism_data_cleaned.csv
    
    # 1. Outlier Capping
    for col in ['DurationOfPitch', 'NumberOfTrips', 'MonthlyIncome']:
        if col in df.columns:
            upper_limit = df[col].quantile(0.99)
            df[col] = df[col].clip(upper=upper_limit)
    
    # 2. Advanced Interaction Engineering
    if 'NumberOfPersonVisiting' in df.columns and 'NumberOfChildrenVisiting' in df.columns:
        df['Adults'] = df['NumberOfPersonVisiting'] - df['NumberOfChildrenVisiting']
    
    if 'MonthlyIncome' in df.columns and 'Adults' in df.columns:
        df['IncomePerPerson'] = df['MonthlyIncome'] / (df['Adults'] + 1)
        if 'Age' in df.columns:
            df['Income_to_Age_Ratio'] = df['MonthlyIncome'] / (df['Age'])
            df['Income_Seniority'] = df['MonthlyIncome'] * df['Age']
    return df
'''
   # 3A. Luxury Alignment (Using String-based keys)
    desig_tier_map = {'Executive': 1, 'Manager': 2, 'Senior Manager': 3, 'AVP': 4, 'VP': 5}
    prod_tier_map = {'Basic': 1, 'Deluxe': 2, 'Standard': 3, 'Super Deluxe': 4, 'King': 5}
    
    if 'Designation' in df.columns:
        df['Designation_Tier'] = df['Designation'].map(desig_tier_map).fillna(2)
    if 'ProductPitched' in df.columns:
        df['Product_Tier'] = df['ProductPitched'].map(prod_tier_map).fillna(2)
        
    if 'Designation_Tier' in df.columns and 'Product_Tier' in df.columns and 'MonthlyIncome' in df.columns:
        df['LuxuryIndex'] = df['Designation_Tier'] * df['Product_Tier'] * (df['MonthlyIncome'] / 1000)
        df['IncomePerTier'] = df['MonthlyIncome'] / (df['Product_Tier'] + 1)
    
    # 3B. Categorical Handling
    # We cast everything to string to ensure pd.get_dummies handles them as categories
    one_hot_cols = ['Occupation', 'ProductPitched', 'MaritalStatus', 'Designation', 'Gender', 'TypeofContact']
    for col in one_hot_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
'''
