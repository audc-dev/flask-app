from flask import Flask, render_template, request
import joblib
import numpy as np
import os
import logging
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load your trained model
def load_model():
    model_path = os.path.join('model', 'best_churn_model.joblib')
    try:
        with open(model_path, 'rb') as file:
            return joblib.load(file)
    except FileNotFoundError:
        logger.error("Model file not found. Please ensure 'best_churn_model.joblib' is in the 'model/' directory.")
        return None
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None

# Load Excel Data
def load_customer_data():
    try:
        # Print current working directory
        logger.info("Current Working Directory: %s", os.getcwd())
        
        # Use absolute path
        excel_path = os.path.abspath('database/customers.xlsx')
        logger.info("Full Excel Path: %s", excel_path)
        
        # Check if file exists
        if not os.path.exists(excel_path):
            logger.error(f"Excel file not found at {excel_path}")
            return None
        
        # Try reading the file with error handling
        try:
            df = pd.read_excel(excel_path, sheet_name='telco_churn')
            # Clean column names (remove leading/trailing spaces)
            df.columns = df.columns.str.strip()
        except Exception as read_error:
            logger.error(f"Error reading Excel file: {read_error}")
            return None
        
        # Validate required columns
        required_columns = [
            'Customer ID',
            'Age',
            'Monthly Charge',
            'Total Extra Data Charges',
            'Total Charges',
            'Internet Service',
            'Avg Monthly GB Download',
            'Avg Monthly Long Distance Charges',
            'Population',
            'Satisfaction Score',
            'Tenure in Months',
            'Total Long Distance Charges',
            'Total Revenue',
            'Churn Category'
        ]
        
        # Check if all required columns exist
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing columns in Excel file: {missing_columns}")
            logger.error(f"Actual columns in file: {df.columns.tolist()}")
            return None
        
        return df
    except Exception as e:
        logger.error(f"Unexpected error loading customer data: {e}")
        return None

# Load model and customer data when the application starts
model = load_model()
customer_data = load_customer_data()

@app.route('/')
@app.route('/index')
def index():
    # If customer data is loaded, get list of customer IDs
    if customer_data is not None:
        customer_ids = customer_data['Customer ID'].tolist()
        return render_template('index.html', customer_ids=customer_ids)
    else:
        logger.error("Customer data could not be loaded")
        return "Error: Could not load customer data. Check the logs for more information.", 500

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # Check if model is loaded
    if model is None:
        return "Error: Machine Learning Model not loaded.", 500
    
    if customer_data is None:
        return "Error: Customer data not loaded.", 500

    # Handle GET request - show customer selection
    if request.method == 'GET':
        customer_ids = customer_data['Customer ID'].tolist()
        return render_template('predict.html', customer_ids=customer_ids)
    
    # Handle POST request (make prediction)
    try:
        # Get selected customer ID
        customer_id = request.form.get('customer_id')
        if not customer_id:
            return "No customer selected", 400
        
        # Fetch customer data
        customer_row = customer_data[customer_data['Customer ID'] == customer_id]
        if customer_row.empty:
            return "Customer not found", 404
        
        # Prepare input data for prediction
        features_for_prediction = [
            'Monthly Charge', 
            'Total Charges',
            'Total Extra Data Charges',
            'Total Long Distance Charges',
            'Tenure in Months',
            'Avg Monthly GB Download',
            'Avg Monthly Long Distance Charges',
            'Population',
            'Satisfaction Score',
            'Internet Service'  # Only if it wasn't dropped during training
            ]
    
        # Create input DataFrame with ONLY these columns
        input_data = customer_row[features_for_prediction]
    
        # Drop any columns that were dropped during training
        # These should match the cols_to_drop in train.py
        cols_to_drop = [
            'Churn Category', 'Churn Reason', 'Churn Score', 'City', 'CLTV',
            'Contract', 'Country', 'Customer ID', 'Customer Status', 'Dependents',
            'Device Protection Plan', 'Gender', 'Internet Type', 'Lat Long',
            'Latitude', 'Longitude', 'Married', 'Multiple Lines',
            'Number of Dependents', 'Number of Referrals', 'Offer',
            'Online Backup', 'Online Security', 'Paperless Billing', 'Partner',
            'Payment Method', 'Phone Service', 'Premium Tech Support', 'Quarter',
            'Referred a Friend', 'Senior Citizen', 'Total Refunds', 'State',
            'Streaming Movies', 'Streaming Music', 'Streaming TV', 'Under 30',
            'Unlimited Data', 'Zip Code'
            ]
    
        # Ensure we're not trying to use any dropped columns
        input_data = input_data.drop([col for col in cols_to_drop if col in input_data.columns], axis=1)
        
        # Create input DataFrame
        input_data = customer_row[features_for_prediction]
        
        # Preprocess data
        # Ensure categorical variables are properly encoded
        input_data_processed = pd.get_dummies(input_data, 
            columns=['Internet Service'])
        
        # Make sure all expected columns are present (add missing ones with 0)
        expected_columns = [
            'Age',
            'Monthly Charge',
            'Avg Monthly GB Download',
            'Avg Monthly Long Distance Charges',
            'Total Long Distance Charges',
            'Total Extra Data Charges',
            'Total Revenue',
            'Total Charges',
            'Population',
            'Tenure in Months',
            'Satisfaction Score',
            'Internet Service_0',
            'Internet Service_1'
        ]
        
        for col in expected_columns:
            if col not in input_data_processed.columns:
                input_data_processed[col] = 0
        
        # Reorder columns to match model expectations
        input_data_processed = input_data_processed[expected_columns]
        
        # Make prediction
        prediction = model.predict(input_data_processed)[0]
        prediction_proba = model.predict_proba(input_data_processed)[0][1]
        
        # Prepare additional customer details for display
        customer_details = customer_row.iloc[0].to_dict()
        
        # Interpret prediction
        churn_status = "High Risk of Churning" if prediction == 1 else "Low Risk of Churning"
        
        # Render results
        return render_template('predict.html', 
                            customer_id=customer_id,
                            prediction=prediction,
                            prediction_proba=round(prediction_proba, 4),
                            churn_status=churn_status,
                            customer_details=customer_details,
                            monthly_charge=customer_details.get('Monthly Charge', 'N/A'),
                            avg_monthly_gb=customer_details.get('Avg Monthly GB Download', 'N/A'),
                            total_extra_data_charges=customer_details.get('Total Extra Data Charges', 'N/A'),
                            contract_type=customer_details.get('Contract', 'N/A'),
                            internet_service=customer_details.get('Internet Service', 'N/A'),
                            payment_method=customer_details.get('Payment Method', 'N/A'))
    
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return f"An unexpected error occurred during prediction: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True)