import pickle as pk
import random
import warnings

import pandas as pd
from math import floor
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures

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
results['full'] = model_evaluation(X_train, y_train, X_test, y_test, X, y, folds, n_attributes, skip=False)

# Uni-variate filter selection

selector_25 = SelectKBest(f_classif, k=floor(X_train.shape[1]*0.25))
selector_25.fit(X_train, y_train)

X_train_reduced = selector_25.transform(X_train)
X_test_reduced = selector_25.transform(X_test)
X_reduced = selector_25.transform(X)
n_attributes = X_train_reduced.shape[1]

results['f25u'] = model_evaluation(X_train_reduced, y_train, X_test_reduced, y_test, X_reduced, y, folds, n_attributes, skip=False)

selector_50 = SelectKBest(f_classif, k=floor(X_train.shape[1]*0.5))
selector_50.fit(X_train, y_train)

X_train_reduced = selector_50.transform(X_train)
X_test_reduced = selector_50.transform(X_test)
X_reduced = selector_50.transform(X)
n_attributes = X_train_reduced.shape[1]

results['f50u'] = model_evaluation(X_train_reduced, y_train, X_test_reduced, y_test, X_reduced, y, folds, n_attributes, skip=False)

selector_75 = SelectKBest(f_classif, k=floor(X_train.shape[1]*0.75))
selector_75.fit(X_train, y_train)

X_train_reduced = selector_75.transform(X_train)
X_test_reduced = selector_75.transform(X_test)
X_reduced = selector_75.transform(X)
n_attributes = X_train_reduced.shape[1]

results['f75u'] = model_evaluation(X_train_reduced, y_train, X_test_reduced, y_test, X_reduced, y, folds, n_attributes, skip=False)

# Wrapper selection
log_reg = LogisticRegression(penalty="none", solver="saga")
sfsl = SequentialFeatureSelector(log_reg, k_features=floor(X_train.shape[1]*0.25),
                                 forward=True, verbose=1, cv=10, scoring='accuracy')
sfsl = sfsl.fit(X_train, y_train)

X_train_reduced = X_train[:, sfsl.k_feature_idx_]
X_test_reduced = X_test[:, sfsl.k_feature_idx_]
X_reduced = X[:, sfsl.k_feature_idx_]
n_attributes = X_train_reduced.shape[1]

results['w25u'] = model_evaluation(X_train_reduced, y_train, X_test_reduced, y_test, X_reduced, y, folds, n_attributes, skip=False)

sfsl = SequentialFeatureSelector(log_reg, k_features=floor(X_train.shape[1]*0.5),
                                 forward=True, verbose=1, cv=10, scoring='accuracy')
sfsl = sfsl.fit(X_train, y_train)

X_train_reduced = X_train[:, sfsl.k_feature_idx_]
X_test_reduced = X_test[:, sfsl.k_feature_idx_]
X_reduced = X[:, sfsl.k_feature_idx_]
n_attributes = X_train_reduced.shape[1]

results['w50u'] = model_evaluation(X_train_reduced, y_train, X_test_reduced, y_test, X_reduced, y, folds, n_attributes, skip=False)

sfsl = SequentialFeatureSelector(log_reg, k_features=floor(X_train.shape[1]*0.75),
                                 forward=True, verbose=1, cv=10, scoring='accuracy')
sfsl = sfsl.fit(X_train, y_train)

X_train_reduced = X_train[:, sfsl.k_feature_idx_]
X_test_reduced = X_test[:, sfsl.k_feature_idx_]
X_reduced = X[:, sfsl.k_feature_idx_]
n_attributes = X_train_reduced.shape[1]

results['w75u'] = model_evaluation(X_train_reduced, y_train, X_test_reduced, y_test, X_reduced, y, folds, n_attributes, skip=False)

# Multivariate filter and wrapper
# First we create the interaction features and then check if they are good to use
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)
X_poly = poly.transform(X)

# Now we filter
selector_25 = SelectKBest(f_classif, k=floor(X_train.shape[1]*0.25))
selector_25.fit(X_train_poly, y_train)

X_train_reduced = selector_25.transform(X_train_poly)
X_test_reduced = selector_25.transform(X_test_poly)
X_reduced = selector_25.transform(X_poly)
n_attributes = X_train_reduced.shape[1]


results['f25m'] = model_evaluation(X_train_reduced, y_train, X_test_reduced, y_test, X_reduced, y, folds, n_attributes, skip=False)

selector_50 = SelectKBest(f_classif, k=floor(X_train.shape[1]*0.5))
selector_50.fit(X_train_poly, y_train)

X_train_reduced = selector_50.transform(X_train_poly)
X_test_reduced = selector_50.transform(X_test_poly)
X_reduced = selector_50.transform(X_poly)
n_attributes = X_train_reduced.shape[1]

results['f50m'] = model_evaluation(X_train_reduced, y_train, X_test_reduced, y_test, X_reduced, y, folds, n_attributes, skip=False)

selector_75 = SelectKBest(f_classif, k=floor(X_train.shape[1]*0.75))
selector_75.fit(X_train_poly, y_train)

X_train_reduced = selector_75.transform(X_train_poly)
X_test_reduced = selector_75.transform(X_test_poly)
X_reduced = selector_75.transform(X_poly)
n_attributes = X_train_reduced.shape[1]

results['f75m'] = model_evaluation(X_train_reduced, y_train, X_test_reduced, y_test, X_reduced, y, folds, n_attributes, skip=False)

# Wrapper selection
log_reg = LogisticRegression(penalty="none", solver="saga")
sfsl = SequentialFeatureSelector(log_reg, k_features=floor(X_train.shape[1]*0.25),
                                 forward=True, verbose=1, cv=10, scoring='accuracy')
sfsl = sfsl.fit(X_train_poly, y_train)

X_train_reduced = X_train_poly[:, sfsl.k_feature_idx_]
X_test_reduced = X_test_poly[:, sfsl.k_feature_idx_]
X_reduced = X_poly[:, sfsl.k_feature_idx_]
n_attributes = X_train_reduced.shape[1]

results['w25m'] = model_evaluation(X_train_reduced, y_train, X_test_reduced, y_test, X_reduced, y, folds, n_attributes, skip=False)

sfsl = SequentialFeatureSelector(log_reg, k_features=floor(X_train.shape[1]*0.5),
                                 forward=True, verbose=1, cv=10, scoring='accuracy')
sfsl = sfsl.fit(X_train_poly, y_train)

X_train_reduced = X_train_poly[:, sfsl.k_feature_idx_]
X_test_reduced = X_test_poly[:, sfsl.k_feature_idx_]
X_reduced = X_poly[:, sfsl.k_feature_idx_]
n_attributes = X_train_reduced.shape[1]

results['w50m'] = model_evaluation(X_train_reduced, y_train, X_test_reduced, y_test, X_reduced, y, folds, n_attributes, skip=False)

sfsl = SequentialFeatureSelector(log_reg, k_features=floor(X_train.shape[1]*0.75),
                                 forward=True, verbose=1, cv=10, scoring='accuracy')
sfsl = sfsl.fit(X_train_poly, y_train)

X_train_reduced = X_train_poly[:, sfsl.k_feature_idx_]
X_test_reduced = X_test_poly[:, sfsl.k_feature_idx_]
X_reduced = X_poly[:, sfsl.k_feature_idx_]
n_attributes = X_train_reduced.shape[1]

results['w75m'] = model_evaluation(X_train_reduced, y_train, X_test_reduced, y_test, X_reduced, y, folds, n_attributes, skip=False)

file = open("results.pkl", 'wb')
pk.dump(results, file)
file.close()
