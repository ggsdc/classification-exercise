import pickle as pk
import random
import warnings

import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler

from src.functions.model_evaluation import model_evaluation
from src.functions.prepare_data import prepare_data

# from src.functions.rule_induction import rule_induction

warnings.filterwarnings("ignore")

# TODO: Rule induction
# TODO: clustering

random.seed(9)

pd.set_option('display.max_columns', 210)

data = pd.read_csv('./data/train.csv')
data = data.drop(columns=['customerID'])

data = prepare_data(data)

values = data.values
scale = MinMaxScaler()
data_scaled = scale.fit_transform(values)
X = data_scaled[:, 0:-1]
y = data_scaled[:, -1]
y = y.astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

folds = StratifiedKFold(n_splits=10, shuffle=False)
n_attributes = X_train.shape[1]
results = dict()

results['full'] = dict()
results['full'] = model_evaluation(X_train, y_train, X_test, y_test, folds, n_attributes, skip=True)

# TODO: variable selection
# Uni-variate variable selection

selector_25 = SelectKBest(f_classif, k=6)
selector_25.fit(X_train, y_train)

X_train_reduced = selector_25.transform(X_train)
X_test_reduced = selector_25.transform(X_test)
n_attributes = X_train_reduced.shape[1]

results['25'] = model_evaluation(X_train_reduced, y_train, X_test_reduced, y_test, folds, n_attributes, skip=True)

selector_50 = SelectKBest(f_classif, k=13)
selector_50.fit(X_train, y_train)

X_train_reduced = selector_50.transform(X_train)
X_test_reduced = selector_50.transform(X_test)
n_attributes = X_train_reduced.shape[1]

results['50'] = model_evaluation(X_train_reduced, y_train, X_test_reduced, y_test, folds, n_attributes, skip=True)

selector_75 = SelectKBest(f_classif, k=19)
selector_75.fit(X_train, y_train)

X_train_reduced = selector_75.transform(X_train)
X_test_reduced = selector_75.transform(X_test)
n_attributes = X_train_reduced.shape[1]

results['75'] = model_evaluation(X_train_reduced, y_train, X_test_reduced, y_test, folds, n_attributes, skip=True)

file = open("results.pkl", 'wb')
pk.dump(results, file)
file.close()

# TODO: model looping.
# TODO: model selection.
