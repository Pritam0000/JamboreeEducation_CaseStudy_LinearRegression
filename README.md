# Jamboree Admission Prediction

This project predicts the chance of admission to Ivy League colleges for Indian students using machine learning techniques.

## Setup

1. Clone this repository.
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
3. Place the `jamboree_admission.csv` file in the `data/` directory.

## Usage

1. Run the Streamlit app:
   ```
   streamlit run app.py
   ```
2. Use the web interface to make predictions or retrain the model.

## Project Structure

- `data/`: Contains the dataset
- `models/`: Stores the trained model
- `src/`: Contains source code for data preprocessing, model training, and evaluation
- `app.py`: Main Streamlit application
- `requirements.txt`: List of required Python packages

## Retraining the Model

You can retrain the model using the web interface. Adjust the parameters as needed and click the "Retrain Model" button.