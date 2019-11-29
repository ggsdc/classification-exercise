import pandas as pd
import random
from src.functions.KNN_classifier import knn_classifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler

random.seed(9)

pd.set_option('display.max_columns', 210)

data = pd.read_csv('./data/train.csv')
data = data.drop(columns=['ID_code'])
values = data.values
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(values)
X = data_scaled[:, 1:]
y = data_scaled[:, 0:1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

folds = StratifiedKFold(n_splits=10, shuffle=False)

results = knn_classifier(X_train, y_train, X_test, y_test, folds)





# TODO: cross-validation?
# TODO: model looping.
# TODO: model selection.