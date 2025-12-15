import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingRegressor

df = pd.read_csv('./data/sample_movies.csv')

X = df.drop('success_score', axis=1)
y = df['success_score']

model = GradientBoostingRegressor()
model.fit(X, y)

joblib.dump(model, 'greenlight_model.joblib')

