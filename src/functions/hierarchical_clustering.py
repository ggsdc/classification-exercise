import itertools as it

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import confusion_matrix, classification_report, adjusted_rand_score


def hierarchical_clustering(x_train, y_train, x, y):
    print("HAC")
    n_cl = 2
    affinity = ['euclidean', 'manhattan']
    linkages = ['average', 'complete', 'single']
    results = dict()
    for a, l in it.product(affinity, linkages):
        model = AgglomerativeClustering(n_clusters=n_cl, affinity=a, linkage=l)
        model.fit(x_train)
        results[(a, l)] = adjusted_rand_score(y_train, model.labels_)
        print((a, l), results[(a, l)])

    best_acc = -1.1
    best_config = tuple()
    for i in results:
        if results[i] > best_acc:
            best_acc = results[i]
            best_config = i

    best_model = AgglomerativeClustering(n_clusters=n_cl, affinity=best_config[0], linkage=best_config[1])
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

    return True
