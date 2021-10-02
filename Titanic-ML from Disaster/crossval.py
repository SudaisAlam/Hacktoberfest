import pandas as pd

# Read the data
data = pd.read_csv('/Users/sudaisalam/Downloads/titanic/train.csv')

# Select subset of predictors
cols_to_use =  ['Pclass', 'Age', 'Fare']
X = data[cols_to_use]

# Select target
y = data.Survived
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

my_pipeline = Pipeline(
    steps=[
        ('preprocessor', SimpleImputer()),
        ('model', RandomForestRegressor(n_estimators=50, random_state=0))
        ]
)

from sklearn.model_selection import cross_val_score

# Multiply by -1 since sklearn calculates *negative* MAE
scores = -1 * cross_val_score(my_pipeline, X, y, cv=5, scoring='neg_mean_absolute_error')

print("MAE scores:\n", scores)