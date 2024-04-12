import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

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
categorical_cols = [cname for cname in X_train_f.columns if X_train_f[cname].nunique() < 10 and 
                        X_train_f[cname].dtype == "object"]

#select numerical data
numerical_cols = [cname for cname in X_train_f.columns if X_train_f[cname].dtype in ['int64', 'float64']]

cols = categorical_cols + numerical_cols
X_train = X_train_f[cols].copy()
X_valid = X_valid_f[cols].copy()
X_test = X_test_full[cols].copy()

#preprocessing numerical data
numerical_transformer = SimpleImputer(strategy='constant')

#preproccesing categorical data
categorical_transformer  = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

#bundle preprocessing for numerical ant categorical data    
preproccesor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

#define model
model = RandomForestRegressor(n_estimators=100, random_state=1)

#bundle preprocessing and modeling code in pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preproccesor),
                              ('model', model)
                              ])

# preprocessing of training data 
my_pipeline.fit(X_train,y_train)

preds = my_pipeline.predict(X_valid)
mae = mean_absolute_error(preds, y_valid)

print(mae)

#predict

predictions = my_pipeline.predict(X_test)
output = pd.DataFrame({'Id': X_test.index,
                       'SalePrice': predictions})
output.to_csv('submission_Pipeline.csv', index=False)



