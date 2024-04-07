import pandas as pd
from sklearn.model_selection import train_test_split

X_train = pd.read_csv('data/train.csv', index_col='Id')
X_Test = pd.read_csv('data/test.csv', index_col='Id')



