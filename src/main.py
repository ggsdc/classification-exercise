import random
import warnings

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler

from src.functions.KNN_classifier import knn_classifier
from src.functions.classification_trees import decision_trees
from src.functions.logistic_regression import logistic_regression
from src.functions.metaclassifiers import bagging, random_forest, gradient_boosting
from src.functions.naive_bayes import naive_bayes
from src.functions.prepare_data import prepare_data
from src.functions.rule_induction import rule_induction

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
results['knn'] = knn_classifier(X_train, y_train, folds, n_attributes)
results['decision-trees'] = decision_trees(X_train, y_train, X_test, y_test, folds)
results['naive-bayes'] = naive_bayes(X_train, y_train, folds)
results['rule-induction'] = rule_induction(X_train, y_train, folds)
results['logistic-regression'] = logistic_regression(X_train, y_train, X_test, y_test, folds)
results['bagging'] = bagging(X_train, y_train, X_test, y_test, folds)
results['random-forest'] = random_forest(X_train, y_train, X_test, y_test, folds)
results['gradient-boosting'] = gradient_boosting(X_train, y_train, X_test, y_test, folds)

# TODO: variable selection
# TODO: model looping.
# TODO: model selection.
