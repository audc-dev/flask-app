import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def preprocess_customer_data(input_file, output_file):
    """
    Load customer data from Excel, apply preprocessing, and save to new Excel file.
    
    Args:
        input_file (str): Path to input Excel file
        output_file (str): Path to save processed Excel file
    """
    # Load the data
    df = pd.read_excel(input_file, sheet_name='customers')
    
    print(f"Original data shape: {df.shape}")
    
    # List of columns to remove (from train.py)
    COLS_TO_DROP = [
        'Lat Long', 'Latitude', 'Longitude', 'Zip Code', 
        'City', 'State', 'Country', 'Quarter', 'Churn Category', 
        'Churn Reason', 'Churn Score', 'Category', 'Customer Status', 
        'Dependents', 'Device Protection Plan', 'Gender', 'Under 30', 
        'Married', 'Number of Dependents', 'Number of Referrals',
        'Payment Method', 'Offer', 'Online Backup', 'Online Security', 
        'Paperless Billing', 'Partner', 'Premium Tech Support', 
        'Referred a Friend', 'Senior Citizen', 'Total Refunds'
    ]
    
    # Safely remove columns (only those present in dataframe)
    df = df.drop([col for col in COLS_TO_DROP if col in df.columns], axis=1)
    print(f"After dropping columns: {df.shape}")
    
    # Convert Total Charges to numeric (handling non-numeric values)
    df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')
    
    # Handle missing values in numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"Imputed missing values in {col} with median: {median_val}")
    
    # Identify feature types for preprocessing
    # Note: We're not doing the actual model preprocessing, just preparing the data
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    print("\nFeature Breakdown:")
    print(f"Categorical: {cat_cols}")
    print(f"Numeric: {num_cols}")
    
    # Save the processed data
    df.to_excel(output_file, index=False)
    print(f"\nProcessed data saved to {output_file}")
    print(f"Final data shape: {df.shape}")

# Example usage
if __name__ == "__main__":
    input_excel = "database/customers.xlsx"
    output_excel = "database/customers_processed.xlsx"
    preprocess_customer_data(input_excel, output_excel)