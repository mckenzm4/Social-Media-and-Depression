from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd

#neural network configuration
def build_nn(input_dim):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=input_dim))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))  # output for regression
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

#train linear regression, svr, xgboost, and neural network models
#return dictionary of trained models
def train_models(X_train, y_train):
    
    models = {}

    # Convert sparse matrix to dense once for those that need it
    X_dense = X_train.toarray() if hasattr(X_train, "toarray") else X_train

    # Linear Regression
    start_time = time.time()
    lr = LinearRegression()
    lr.fit(X_dense, y_train)
    models['lr_model'] = lr
    print(f"Linear Regression training time: {time.time() - start_time:.4f} seconds")

    # Support Vector Regression
    start_time = time.time()
    svr = SVR(kernel='rbf', C=1.0, epsilon=0.2)
    svr.fit(X_dense, y_train)
    models['svr_model'] = svr
    print(f"Support Vector Regression training time: {time.time() - start_time:.4f} seconds")

    # XGBoost (can handle sparse input)
    start_time = time.time()
    xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    xgb.fit(X_train, y_train)
    models['xgb_model'] = xgb
    print(f"XGB Regressor training time: {time.time() - start_time:.4f} seconds")

    # Neural Network (needs dense input)
    start_time = time.time()
    nn = build_nn(input_dim=X_dense.shape[1])
    nn.fit(X_dense, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)
    models['nn_model'] = nn
    print(f"Sequential Neural Network training time: {time.time() - start_time:.4f} seconds")

    return models

#Calculate regression evaluation metrics
def evaluate_regression(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }

#Evaluate all models
def evaluate_all(models: dict, X_test, y_test):
    import pandas as pd

    results = {}
    X_dense = X_test.toarray() if hasattr(X_test, "toarray") else X_test

    for name, model in models.items():
        if hasattr(model, 'predict'):
            # Models that require dense input
            if any(key in name for key in ['lr', 'svr', 'nn']):
                X_eval = X_dense
            else:
                X_eval = X_test  # e.g., XGBoost can handle sparse

            y_pred = model.predict(X_eval)
            results[name] = evaluate_regression(y_test, y_pred)

    return pd.DataFrame(results).T
