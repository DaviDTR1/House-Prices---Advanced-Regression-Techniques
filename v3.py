import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt


#read data
X_full = pd.read_csv('data/train.csv', index_col='Id')
X_test_full = pd.read_csv('data/test.csv', index_col='Id')

#remove row with missing target
X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X_full.SalePrice
X_full.drop(['SalePrice'],axis=1,inplace=True)

#split test validation
X_train_f, X_valid_f, y_train, y_valid = train_test_split(X_full, y, train_size = 0.8, test_size = 0.2, random_state = 0)

#select categorical data 
categorical_cols = [cname for cname in X_full.columns if X_train_f[cname].nunique() < 10 and 
                        X_full[cname].dtype == "object"]

#select numerical data
numerical_cols = [cname for cname in X_full.columns if X_full[cname].dtype in ['int64', 'float64']]

cols = categorical_cols + numerical_cols
X_train = X_train_f[cols].copy()
X_valid = X_valid_f[cols].copy()
X_test = X_test_full[cols].copy()
X = X_full[cols].copy()

#one hot
X_train = pd.get_dummies(X_train)
X_valid = pd.get_dummies(X_valid)
X_test = pd.get_dummies(X_test)
X = pd.get_dummies(X)

X_train, X_valid = X_train.align(X_valid, join='left', axis=1)
X_train, X_test = X_train.align(X_test, join='left', axis=1)
X, X_test = X.align(X_test, join='left', axis=1)

#model
model = XGBRegressor(n_estimators = 500, early_stopping_rounds = 5,
                          learning_rate = 0.05, n_jobs=5)

model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)

#train model
model.fit(X, y, eval_set=[(X, y)], verbose=False)

predictions = model.predict(X_test)
output = pd.DataFrame({'Id': X_test.index,
                       'SalePrice': predictions})
output.to_csv('submission_XGboost.csv', index=False)