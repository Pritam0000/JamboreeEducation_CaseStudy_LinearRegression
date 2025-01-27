from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2
    }

def predict_admission_chance(model, input_data):
    return model.predict(input_data)[0]