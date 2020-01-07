import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier


# TODO: check the distance to be used in class slides
# TODO: try with different weights and distances to get different results -> Brute force
# TODO: brute force number of neighbors.

def knn_classifier_wo_folds(x_train, y_train, x_test, y_test, n_attributes):
    print("CNN")
    distances = [1, 2]  # Manhattan and euclidean
    results = dict()

    for d in distances:
        results[d] = dict()

        loop = True
        # HERE WE APPLY THE LOGIC FOR THE CNN
        # WE ARE GOING TO MODIFY THE LOGIC SLIGHTLY.
        # WE START WITH THE FIRST SAMPLE DIRECTLY.
        order_idx = np.delete(np.arange(x_train.shape[0]), 0)
        x_store = x_train[0]
        y_store = y_train[0]
        x_order = x_train[order_idx]
        y_order = y_train[order_idx]
        loop_count = 1

        model = KNeighborsClassifier(n_neighbors=1, weights='uniform', algorithm='kd_tree', p=d)

        while loop:
            model = KNeighborsClassifier(n_neighbors=1, weights='uniform', algorithm='kd_tree', p=d)
            model.fit(x_store.reshape(loop_count, n_attributes), y_store.reshape(loop_count, 1))
            finished = True
            i = 0
            for i in range(x_order.shape[0]):
                predicted = model.predict(x_order[i].reshape(1, n_attributes))
                if predicted != y_order[i]:
                    x_store = np.concatenate(
                        (x_store.reshape(loop_count, n_attributes), x_order[i].reshape(1, n_attributes)))
                    y_store = np.concatenate((y_store.reshape(loop_count, 1), y_order[i].reshape(1, 1)))

                    order_idx = np.delete(order_idx, i)

                    x_order = x_train[order_idx]
                    y_order = y_train[order_idx]

                    finished = False
                    break

            if loop_count % 100 == 0:
                print(loop_count, 'iterations')
                print((d, i), x_order.shape, y_order.shape, x_store.shape, y_store.shape)

            loop_count += 1
            if finished:
                print('Finished fold')
                print(model, x_order.shape, y_order.shape, x_store.shape, y_store.shape)
                loop = False

        results[d]['acc'] = model.score(x_test, y_test)
        results[d]['x_order'] = x_order
        results[d]['y_order'] = y_order
        results[d]['x_store'] = x_store
        results[d]['y_store'] = y_store

        print(d, results[d]['acc'])

    best_acc = 0
    best_config = ''
    for i in results:
        total_acc = results[i]['acc']
        if total_acc > best_acc:
            best_acc = total_acc
            best_config = i

    print(best_config, best_acc)
    x_store = results[best_config]['x_store']
    y_store = results[best_config]['y_store']

    best_model = KNeighborsClassifier(n_neighbors=1, weights='uniform', algorithm='kd_tree', p=best_config)
    best_model.fit(x_store, y_store)

    y_prediction = best_model.predict(x_test)
    cm = confusion_matrix(y_test, y_prediction)
    cr = classification_report(y_test, y_prediction, output_dict=True)

    final_results = dict()
    final_results['folds'] = results
    final_results['model'] = best_model
    final_results['acc'] = best_model.score(x_test, y_test)
    final_results['cm'] = cm
    final_results['cr'] = cr
    print(final_results['acc'])

    return final_results
