import itertools as it

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier


def decision_trees(x_train, y_train, x_test, y_test, folds):
    print("DECISION TREES")
    results = dict()
    criteria = ['gini', 'entropy']
    splitters = ['random', 'best']
    fold = 1
    for c, s in it.product(criteria, splitters):
        results[(c, s)] = dict()
        for idx_train, idx_test in folds.split(x_train, y_train):
            results[(c, s)][fold] = 0
            x_train_folds = x_train[idx_train]
            x_test_folds = x_train[idx_test]
            y_train_folds = y_train[idx_train]
            y_test_folds = y_train[idx_test]

            model = DecisionTreeClassifier(criterion=c, splitter=s)
            model.fit(x_train_folds, y_train_folds)
            results[(c, s)][fold] = model.score(x_test_folds, y_test_folds)
            print((c, s), fold, (results[(c, s)][fold]))
            fold += 1

    best_acc = 0
    best_config = ''
    for i in results:
        total_acc = 0
        for j in results[i]:
            total_acc += results[i][j]
        mean_acc = total_acc / 10
        if mean_acc > best_acc:
            best_acc = mean_acc
            best_config = i

    print(best_config, best_acc)

    best_model = DecisionTreeClassifier(criterion=best_config[0], splitter=best_config[1])
    best_model.fit(x_train, y_train)
    y_prediction = best_model.predict(x_test)
    cm = confusion_matrix(y_test, y_prediction)
    cr = classification_report(y_test, y_prediction, output_dict=True)

    final_results = dict()
    final_results['folds'] = results
    final_results['model'] = best_model
    final_results['acc'] = best_model.score(x_test, y_test)
    final_results['cm'] = cm
    final_results['cr'] = cr

    return final_results
