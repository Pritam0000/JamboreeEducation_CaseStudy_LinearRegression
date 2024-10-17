import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(file_path):
    # Load the data
    df = pd.read_csv(file_path)
    
    # Drop the 'Serial No.' column
    df = df.drop('Serial No.', axis=1)
    
    # Split features and target
    X = df.drop('Chance of Admit ', axis=1)
    y = df['Chance of Admit ']
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, scaler

def prepare_input_data(input_data, scaler):
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Scale the input data
    input_scaled = pd.DataFrame(scaler.transform(input_df), columns=input_df.columns)
    
    return input_scaled