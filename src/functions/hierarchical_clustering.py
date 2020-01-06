import itertools as it

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import confusion_matrix, classification_report


def hierarchical_clustering(x_train, y_train, x, y):
    print("HAC")
    n_cl = 2
    affinity = ['euclidean', 'manhattan']
    linkages = ['average', 'complete', 'single']
    results = dict()
    for a, l in it.product(affinity, linkages):
        model = AgglomerativeClustering(n_clusters=n_cl, affinity=a, linkage=l)
        model.fit(x_train)
        cm = confusion_matrix(y_train, model.labels_)
        aux = (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1])
        if aux > 0.5:
            results[(a, l)] = aux
        else:
            results[(a, l)] = 1 - aux
        print((a, l), results[(a, l)])

    best_acc = 0
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
    print(final_results['acc'])
    final_results['cm'] = cm
    final_results['cr'] = cr

    return True
