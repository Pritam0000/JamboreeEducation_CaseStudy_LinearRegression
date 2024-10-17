from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import GridSearchCV
import joblib

def train_model(X_train, y_train, model_type='linear', params=None):
    if model_type == 'linear':
        model = LinearRegression()
    elif model_type == 'ridge':
        model = Ridge()
    elif model_type == 'lasso':
        model = Lasso()
    else:
        raise ValueError("Invalid model type. Choose 'linear', 'ridge', or 'lasso'.")
    
    if params:
        grid_search = GridSearchCV(model, params, cv=5)
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_
    else:
        model.fit(X_train, y_train)
    
    return model

def save_model(model, file_path):
    joblib.dump(model, file_path)

def load_model(file_path):
    return joblib.load(file_path)