import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# 1. Load Data
df = pd.read_csv("D:/1.UNI 5th Semester/6. Advance DataBase/ADVANCE_DB PROJECT/ds_project/mellyapp/ml/mellyapp_dataset_OLAP.csv")
X = df.drop('grade', axis=1)
y = df['grade']

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Pipeline
pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('model', XGBRegressor(objective='reg:squarederror', random_state=42))
])

# 4. Grid Search (opsional → bisa langsung pakai default jika mau simple)
param_grid = {
    'model__n_estimators': [100],
    'model__max_depth': [3],
    'model__learning_rate': [0.1]
}

grid = GridSearchCV(pipe, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
grid.fit(X_train, y_train)

# 5. Evaluasi
y_pred = grid.predict(X_test)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
print(f"XGBoost RMSE: {rmse:.2f}")

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'xgboost_model.pkl')

joblib.dump(grid.best_estimator_, model_path)

