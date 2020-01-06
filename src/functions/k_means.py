from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, classification_report


def k_means(x_train, y_train, x_test, y_test, folds):
    n_cl = 2
    initialization = ['random', 'k-means++']
    results = dict()
    for i in initialization:
        results[i] = dict()
        fold = 1
        for idx_train, idx_test in folds.split(x_train, y_train):
            x_train_folds = x_train[idx_train]
            x_test_folds = x_train[idx_test]
            y_test_folds = y_train[idx_test]
            model = KMeans(n_clusters=n_cl, init=i)
            model.fit(x_train_folds)
            cm = confusion_matrix(y_test_folds, model.predict(x_test_folds))
            aux = (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1])
            if aux > 0.5:
                results[i][fold] = aux
            else:
                results[i][fold] = 1 - aux
            print((i, fold), results[i][fold])
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

    best_model = KMeans(n_clusters=n_cl, init=best_config)
    best_model.fit(x_train)
    cm = confusion_matrix(y_test, best_model.predict(x_test))
    cr = classification_report(y_test, best_model.predict(x_test), output_dict=True)

    final_results = dict()
    final_results['folds'] = results
    final_results['model'] = best_model
    aux = (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1])
    if aux > 0.5:
        final_results['acc'] = aux
    else:
        final_results['acc'] = 1 - aux
    print(final_results['acc'])
    final_results['cm'] = cm
    final_results['cr'] = cr

    return final_results
