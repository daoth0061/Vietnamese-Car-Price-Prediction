import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, FunctionTransformer
from Ankun_Multicolumn import MultiColumnLabelEncoder

# Import the dataset
dataset = pd.read_csv("D:\Project\Car-Evaluation\Dataset\Final\Remove-null-car_name-and-fill-null.csv")

# Split features and target column
features = ['origin', 'car_model', 'mileage', 'exterior_color', 'interior_color', 'num_of_doors',
            'seating_capacity', 'engine', 'engine_capacity', 'transmission', 'drive_type',
            'fuel_consumption', 'brand', 'grade', 'year_of_manufacture']
target = 'price_in_billion'
X = dataset[features] # X = dataset.iloc[:, :-1]
y = dataset[target] # y = dataset.iloc[:, -1].reshape(-1, 1)

categorical_columns = ['origin', 'car_model', 'exterior_color', 'interior_color',
            'engine', 'transmission', 'drive_type','brand', 'grade']

# Create a ColumnTransformer with the custom MultiColumnLabelEncoder
preprocessor = ColumnTransformer(
    transformers=[
        ('label', MultiColumnLabelEncoder(columns=categorical_columns), categorical_columns)
    ],
    remainder='passthrough'  # This will keep the other columns unchanged
)

# Create a Pipeline with the preprocessor
pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
# Fit and transform the data
X_transformed = pipeline.fit_transform(X)



joblib.dump(pipeline, "Ankun_processing.pkl")
