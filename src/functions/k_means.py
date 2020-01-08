from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, classification_report, adjusted_rand_score


def k_means(x_train, y_train, x, y, folds):
    print('KMEANS')
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
            results[i][fold] = adjusted_rand_score(y_test_folds, model.predict(x_test_folds))
            print((i, fold), results[i][fold])
            fold += 1

    best_acc = -1.1
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
    best_model.fit(x)
    cm = confusion_matrix(y, best_model.labels_)
    cr = classification_report(y, best_model.labels_, output_dict=True)

    final_results = dict()
    final_results['folds'] = results
    final_results['model'] = best_model
    aux = (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1])
    if aux > 0.5:
        final_results['acc'] = aux
    else:
        final_results['acc'] = 1 - aux

    final_results['cm'] = cm
    final_results['cr'] = cr
    final_results['ari'] = adjusted_rand_score(y, best_model.labels_)
    print(final_results['ari'])

    return final_results
