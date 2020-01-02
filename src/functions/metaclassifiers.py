import itertools as it
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, VotingClassifier, StackingClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier

def bagging(X_train, y_train, X_test, y_test, folds, n_attributes):
    print("BAGGING")
    results = dict()

    num_estimators = [10, 25, 50]
    max_samples = [0.5, 0.75, 0.95]
    max_features = [0.5, 0.75, 0.95]
    bootstrap_f = [True, False]
    fold = 1
    for n, s, f, bf in it.product(num_estimators, max_samples, max_features, bootstrap_f):
        results[(n, s, f, bf)] = dict()
        for idx_train, idx_test in folds.split(X_train, y_train):
            results[(n, s, f, bf)][fold] = 0
            X_train_folds = X_train[idx_train]
            X_test_folds = X_train[idx_test]
            y_train_folds = y_train[idx_train]
            y_test_folds = y_train[idx_test]

            model = BaggingClassifier(n_estimators=n, max_samples=s, max_features=f, bootstrap_features=bf, n_jobs=-1)
            model.fit(X_train_folds, y_train_folds)
            results[(n, s, f, bf)][fold] = model.score(X_test_folds, y_test_folds)
            fold += 1

        acc = 0
        for i in results[(n, s, f, bf)]:
            acc += results[(n, s, f, bf)][i]
        acc = acc / 10
        print((n, s, f, bf), acc)


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

    best_model = BaggingClassifier(n_estimators=best_config[0], max_samples=best_config[1], max_features=best_config[2],
                                   bootstrap_features=best_config[3], n_jobs=-1)
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, output_dict=True)

    final_results = dict()
    final_results['folds'] = results
    final_results['model'] = best_model
    final_results['acc'] = best_model.score(X_test, y_test)
    final_results['cm'] = cm
    final_results['cr'] = cr

    return final_results

def random_forest(X_train, y_train, X_test, y_test, folds, n_attributes):
    print("RANDOM FOREST")
    results = dict()

    num_estimators = [100, 200, 300, 500]
    max_features = [0.5, 0.75, 0.95]
    criterions = ['gini', 'entropy']
    fold = 1
    for n, f, c in it.product(num_estimators, max_features, criterions):
        results[(n, f, c)] = dict()
        for idx_train, idx_test in folds.split(X_train, y_train):
            results[(n, f)][fold] = 0
            X_train_folds = X_train[idx_train]
            X_test_folds = X_train[idx_test]
            y_train_folds = y_train[idx_train]
            y_test_folds = y_train[idx_test]

            model = RandomForestClassifier(n_estimators=n, criterion=c, max_features=f, n_jobs=-1)
            model.fit(X_train_folds, y_train_folds)
            results[(n, f, c)][fold] = model.score(X_test_folds, y_test_folds)
            fold += 1

        acc = 0
        for i in results[(n, f, c)]:
            acc += results[(n, f, c)][i]
        acc = acc / 10
        print((n, f, c), acc)

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

    best_model = RandomForestClassifier(n_estimators=best_config[0], max_features=best_config[1],
                                        criterion=best_config[2], n_jobs=-1)
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, output_dict=True)

    final_results = dict()
    final_results['folds'] = results
    final_results['model'] = best_model
    final_results['acc'] = best_model.score(X_test, y_test)
    final_results['cm'] = cm
    final_results['cr'] = cr

    return final_results

def gradient_boosting(X_train, y_train, X_test, y_test, folds, n_attributes):
    print("GRADIENT BOOSTING")
    results = dict()

    num_estimators = [50, 100, 200]
    criterions = ['gini', 'entropy']
    splitters = ['random', 'best']
    fold = 1
    for c, s, n in it.product(criterions, splitters, num_estimators):
        results[(c, s, n)] = dict()
        for idx_train, idx_test in folds.split(X_train, y_train):
            results[(c, s, n)][fold] = 0
            X_train_folds = X_train[idx_train]
            X_test_folds = X_train[idx_test]
            y_train_folds = y_train[idx_train]
            y_test_folds = y_train[idx_test]

            model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(criterion=c, splitter=s), n_estimators=n)
            model.fit(X_train_folds, y_train_folds)
            results[(c, s, n)][fold] = model.score(X_test_folds, y_test_folds)
            fold += 1

        acc = 0
        for i in results[(c, s, n)]:
            acc += results[(c, s, n)][i]
        acc = acc / 10
        print((c, s, n), acc)

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

    best_model = AdaBoostClassifier(
        base_estimator=DecisionTreeClassifier(criterion=best_config[0], splitter=best_config[1]),
        n_estimators=best_config[2])
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, output_dict=True)

    final_results = dict()
    final_results['folds'] = results
    final_results['model'] = best_model
    final_results['acc'] = best_model.score(X_test, y_test)
    final_results['cm'] = cm
    final_results['cr'] = cr

    return final_results

def voting_ensemble():
    return True

def stacked_ensemble():
    return True