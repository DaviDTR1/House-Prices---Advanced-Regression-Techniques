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
def ordinal_encoding(X_train, X_valid):
    objects_col = [col for col in X_train.columns if X_train[col].dtype == 'object']
    good_col = [col for col in objects_col if set(X_valid[col]).issubset(X_train[col])]
    bad_col = list(set(objects_col) - set(good_col))
    
    new_X_train = X_train.drop(bad_col, axis=1)
    new_X_valid = X_valid.drop(bad_col, axis=1)
    ordinal_enc = OrdinalEncoder()
    
    new_X_train[good_col] = ordinal_enc.fit_transform(new_X_train[good_col])
    new_X_valid[good_col] = ordinal_enc.fit_transform(new_X_valid[good_col])
    return new_X_train, new_X_valid

#One-Hot Encoding
def oneHot_encoding(X_train, X_valid):
    objects_col = [col for col in X_train.columns if X_train[col].dtype == 'object']
    low_cardinal_cols = [col for col in objects_col if X_train[col].nunique() < 10]
    high_cadinal_cols = list(set(objects_col) - set(low_cardinal_cols))
    
    oh_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    oh_X_t = pd.DataFrame(oh_encoder.fit_transform(X_train[low_cardinal_cols]))
    oh_X_v = pd.DataFrame(oh_encoder.transform(X_valid[low_cardinal_cols]))
    oh_X_t.index = X_train.index
    oh_X_v.index = X_valid.index
    
    X_train = X_train.drop(objects_col, axis=1)
    X_valid = X_valid.drop(objects_col, axis=1)
    X_train = pd.concat([X_train, oh_X_t], axis=1)
    X_valid = pd.concat([X_valid, oh_X_v], axis=1)
    
    X_train.columns = X_train.columns.astype(str)
    X_valid.columns = X_valid.columns.astype(str)
    return X_train, X_valid   

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

X, X_test = drop_columns_missing_values(X_full, X_test_full)

# # drop dtype = 'object'  
# X = drop_object(X_full)
# X_test = drop_object(X_test_full)

#split the train data for a better validation
X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state = 0)

#deal with missing values
# X_train, X_val = drop_columns_missing_values(X_train=X_train, X_val=X_val)
# X_train, X_val = imputation(X_train=X_train, X_val=X_val)
# X_train, X_val = extend_imputation(X_train=X_train, X_val=X_val)

#only use numerical predictors
# X_train, X_val = ordinal_encoding(X_train, X_val)
X_train, X_val = oneHot_encoding(X_train, X_val)

#columns with missing values
# missing_val_count = (X_train.isnull().sum())
# print(missing_val_count)

#create many model and chose the best
model1 = RandomForestRegressor(n_estimators= 50, random_state=0)
model2 = RandomForestRegressor(n_estimators= 100, random_state=0)
model3 = RandomForestRegressor(n_estimators= 100, criterion='absolute_error', random_state=0)
model4 = RandomForestRegressor(n_estimators= 200, min_samples_split=20, random_state=0)
model5 = RandomForestRegressor(n_estimators= 50, max_depth=7, random_state=0)

models = [model1, model2, model3, model4, model5]

best = -1
model = None

#iterarte for the models and compare the mean to take the best
for m in models:
    temp = best_model(m, X_train, X_val, y_train, y_val)
    print(temp)
    if best == -1 or best > temp:
        best = temp
        model = m

# # ordinal encode training
# X, X_test = ordinal_encoding(X, X_test)
# OneHotEncode
X, X_test = oneHot_encoding(X,X_test)


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
output.to_csv('submission_oneHot_encode.csv', index=False)
