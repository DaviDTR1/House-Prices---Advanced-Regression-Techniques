import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder

#find the mean of each model
def best_model(model , X_train, X_val, y_train, y_val):
    model.fit(X_train, y_train)
    prediction = model.predict(X_val)
    return mean_absolute_error(prediction, y_val)

#Categorical Values
#Ordinal Encoding
def ordinal_encoding(X):
    i = (X.dtypes == 'object')
    objects_col = list(i[i].index)
    
    ordinal_enc = OrdinalEncoder()
    
    X[objects_col] = ordinal_enc.fit_transform(X[objects_col])
    return X[objects_col]

#One-Hot Encoding
def oneHot_encoding(X):
    i = (X.dtypes == 'object')
    objects_col = list(i[i].index)
    
    oh_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    new_X = pd.DataFrame(oh_encoder.fit_transform(X[objects_col]))
    new_X.index = X.index
    
    X = X.drop(objects_col, axis=1)
    X = pd.concat([X, new_X], axis=1)
    
    X.columns = X.columns.astype('str')
    return X    

#drop column
def drop_object(X):
    return X.select_dtypes(exclude=['object'])

#work with missing value
#First aproach : Drop Columns
#drop columns with missing values in the data
def drop_columns_missing_values(X_train, X_val) :
    col_with_missing = [col for col in X_train.keys() if X_train[col].isnull().any()]
    
    new_X_train = X_train.drop(col_with_missing, axis=1)
    new_X_val = X_val.drop(col_with_missing, axis=1)
    
    return new_X_train, new_X_val

#Second aproach : Imputation
#replace missing value with the mean value
def imputation(X_train, X_val):
    my_imputer = SimpleImputer()
    
    new_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
    new_X_val = pd.DataFrame(my_imputer.transform(X_val))
    
    new_X_train.columns = X_train.columns
    new_X_val.columns = X_val.columns
    
    return new_X_train, new_X_val

#Third aproach : Extension to Imputation
def extend_imputation(X_train, X_val):
    col_with_missing = [col for col in X_train.keys() if X_train[col].isnull().any()]

    for col in col_with_missing:
        X_train[col + 'was_missing'] = X_train[col].isnull()
        X_val[col + 'was_missing'] = X_val[col].isnull()
    
    return imputation(X_train=X_train,X_val=X_val)

#read data
X_full = pd.read_csv('data/train.csv', index_col='Id')
X_test_full = pd.read_csv('data/test.csv', index_col='Id')

#remove row with missing target
X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X_full.SalePrice
X_full.drop(['SalePrice'],axis=1,inplace=True)

# #only use numerical predictors
# X = ordinal_encoding(X_full)
# X_test = ordinal_encoding(X_test_full)

X = drop_object(X_full)
X_test = drop_object(X_test_full)


#select the features for the model
# y = X_full.SalePrice
# features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
# X = X_full[features].copy()
# X_test = X_test_full[features].copy()

#split the train data for a better validation
X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state = 0)

#columns with missing values
missing_val_count = (X_train.isnull().sum())

#create many model and chose the best
model1 = RandomForestRegressor(n_estimators= 50, random_state=0)
model2 = RandomForestRegressor(n_estimators= 100, random_state=0)
model3 = RandomForestRegressor(n_estimators= 100, criterion='absolute_error', random_state=0)
model4 = RandomForestRegressor(n_estimators= 200, min_samples_split=20, random_state=0)
model5 = RandomForestRegressor(n_estimators= 50, max_depth=7, random_state=0)

models = [model1, model2, model3, model4, model5]

best = -1
model = None

#deal with missing values
# X_train, X_val = drop_columns_missing_values(X_train=X_train, X_val=X_val)
X_train, X_val = imputation(X_train=X_train, X_val=X_val)
# X_train, X_val = extend_imputation(X_train=X_train, X_val=X_val)

#iterarte for the models and compare the mean to take the best
for m in models:
    temp = best_model(m, X_train, X_val, y_train, y_val)
    print(temp)
    if best == -1 or best > temp:
        best = temp
        model = m



#train the model
imputer = SimpleImputer()
finall_X = pd.DataFrame(imputer.fit_transform(X))
finall_X.columns = X.columns
model.fit(finall_X, y)

#predict
finall_X_test = pd.DataFrame(imputer.fit_transform(X_test))
finall_X_test.columns = X_test.columns
predictions = model.predict(finall_X_test)

#create output archive
output = pd.DataFrame({'Id': X_test.index,
                       'SalePrice': predictions
})
output.to_csv('submission_drop.csv', index=False)
