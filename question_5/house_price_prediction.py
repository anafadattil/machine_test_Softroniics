import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
data = pd.read_csv('house_data.csv')
print("Dataset Info:")
print(data.info())
print("\nFirst 5 rows:")
print(data.head())

X = data.drop('Sale Price', axis=1)
y = data['Sale Price']
numerical_cols = ['Number of Bedrooms', 'Square Footage', 'Age of House', 'Garage Size']
categorical_cols = ['Location']
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),  
    ('scaler', StandardScaler())  
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  
])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])
preprocessor.fit(X)
print("\nPreprocessed feature names:")
print(preprocessor.get_feature_names_out())


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")


# Create the Gradient Boosting Regressor pipeline
gb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor(random_state=42))
])
gb_pipeline.fit(X_train, y_train)
y_pred = gb_pipeline.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nInitial Model Performance:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared Score: {r2:.2f}")


# Hyperparameter tuning
param_grid = {
    'regressor__n_estimators': [50, 100, 200],
    'regressor__learning_rate': [0.01, 0.1, 0.2],
    'regressor__max_depth': [3, 4, 5],
    'regressor__min_samples_split': [2, 5],
    'regressor__min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(
    gb_pipeline,
    param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=1
)
print("\nStarting hyperparameter tuning...")
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
print("\nBest Parameters Found:")
for param, value in best_params.items():
    print(f"{param}: {value}")
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)
mse_best = mean_squared_error(y_test, y_pred_best)
r2_best = r2_score(y_test, y_pred_best)

print("\nBest Model Performance:")
print(f"Mean Squared Error: {mse_best:.2f}")
print(f"R-squared Score: {r2_best:.2f}")


# Final model with best parameters
final_gb_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor(
        n_estimators=best_params['regressor__n_estimators'],
        learning_rate=best_params['regressor__learning_rate'],
        max_depth=best_params['regressor__max_depth'],
        min_samples_split=best_params['regressor__min_samples_split'],
        min_samples_leaf=best_params['regressor__min_samples_leaf'],
        random_state=42
    ))
])
final_gb_model.fit(X_train, y_train)
final_pred = final_gb_model.predict(X_test)
final_mse = mean_squared_error(y_test, final_pred)
final_r2 = r2_score(y_test, final_pred)

print("\nFinal Model Performance:")
print(f"Mean Squared Error: {final_mse:.2f}")
print(f"R-squared Score: {final_r2:.2f}")

# Example prediction
sample_house = pd.DataFrame({
    'Number of Bedrooms': [3],
    'Square Footage': [1800],
    'Location': ['Suburb'],
    'Age of House': [15],
    'Garage Size': [2]
})

predicted_price = final_gb_model.predict(sample_house)
print(f"\nPredicted price for sample house: ${predicted_price[0]:,.2f}")