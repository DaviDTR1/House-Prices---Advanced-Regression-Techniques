import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

def best_model(model , X_train, X_val, y_train, y_val):
    model.fit(X_train, y_train)
    prediction = model.predict(X_val)
    return mean_absolute_error(prediction, y_val)



X_full = pd.read_csv('data/train.csv', index_col='Id')
X_test_full = pd.read_csv('data/test.csv', index_col='Id')


y = X_full.SalePrice
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = X_full[features].copy()
X_test = X_test_full[features].copy()

X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state = 0)

model1 = RandomForestRegressor(n_estimators= 50, random_state=0)
model2 = RandomForestRegressor(n_estimators= 100, random_state=0)
model3 = RandomForestRegressor(n_estimators= 100, criterion='absolute_error', random_state=0)
model4 = RandomForestRegressor(n_estimators= 200, min_samples_split=20, random_state=0)
model5 = RandomForestRegressor(n_estimators= 50, max_depth=7, random_state=0)

models = [model1, model2, model3, model4, model5]

best = -1
model = None

for m in models:
    temp = best_model(m, X_train, X_val, y_train, y_val)
    print(temp)
    if best == -1 or best > temp:
        best = temp
        model = m

model.fit(X, y)

predictions = model.predict(X_test)

output = pd.DataFrame({'Id': X_test.index,
                       'SalePrice': predictions
})
output.to_csv('submission4.csv', index=False)
