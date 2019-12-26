import pandas as pd
import warnings
import random
from src.functions.classification_trees import decision_trees
from src.functions.KNN_classifier import knn_classifier
from src.functions.naive_bayes import naive_bayes
from src.functions.prepare_data import prepare_data
from src.functions.rule_induction import rule_induction
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

random.seed(9)

pd.set_option('display.max_columns', 210)

data = pd.read_csv('./data/train.csv')
data = data.drop(columns=['customerID'])

data = prepare_data(data)

values = data.values
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(values)
X = data_scaled[:, 0:-1]
y = data_scaled[:, -1]
y = y.astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

folds = StratifiedKFold(n_splits=10, shuffle=False)
n_attributes = X_train.shape[1]
# results = knn_classifier(X_train, y_train, X_test, y_test, folds, n_attributes)
# results = decision_trees(X_train, y_train, X_test, y_test, folds, n_attributes)
# results = naive_bayes(X_train, y_train, X_test, y_test, folds, n_attributes)
results = rule_induction(X_train, y_train, X_test, y_test, folds, n_attributes)
# TODO: cross-validation?
# TODO: model looping.
# TODO: model selection.