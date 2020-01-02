import numpy as np
from sklearn.neighbors import KNeighborsClassifier


# TODO: check the distance to be used in class slides
# TODO: try with different weights and distances to get different results -> Brute force
# TODO: brute force number of neighbors.

def knn_classifier(x_train, y_train, folds, n_attributes):
    distances = [1, 2]  # Manhattan and euclidean
    results = dict()

    for d in distances:
        results[d] = dict()
        fold = 1
        for idx_train, idx_test in folds.split(x_train, y_train):
            results[d][fold] = list()
            x_train_folds = x_train[idx_train]
            x_test_folds = x_train[idx_test]
            y_train_folds = y_train[idx_train]
            y_test_folds = y_train[idx_test]
            loop = True
            # HERE WE APPLY THE LOGIC FOR THE CNN
            # WE ARE GOING TO MODIFY THE LOGIC SLIGHTLY.
            # WE START WITH THE FIRST SAMPLE DIRECTLY.
            order_idx = np.delete(np.arange(x_train_folds.shape[0]), 0)
            x_store = x_train_folds[0]
            y_store = y_train_folds[0]
            x_order = x_train_folds[order_idx]
            y_order = y_train_folds[order_idx]
            print(x_order.shape, y_order.shape, x_store.shape, y_store.shape)
            print(y_store)
            loop_count = 1

            model = KNeighborsClassifier(n_neighbors=1, weights='uniform', algorithm='kd_tree', p=d)

            while loop:
                model = KNeighborsClassifier(n_neighbors=1, weights='uniform', algorithm='kd_tree', p=d)
                model.fit(x_store.reshape(loop_count, n_attributes), y_store.reshape(loop_count, 1))
                finished = True
                for i in range(x_order.shape[0]):
                    predicted = model.predict(x_order[i].reshape(1, n_attributes))
                    if predicted != y_order[i]:
                        x_store = np.concatenate(
                            (x_store.reshape(loop_count, n_attributes), x_order[i].reshape(1, n_attributes)))
                        y_store = np.concatenate((y_store.reshape(loop_count, 1), y_order[i].reshape(1, 1)))

                        order_idx = np.delete(order_idx, i)

                        x_order = x_train_folds[order_idx]
                        y_order = y_train_folds[order_idx]

                        print(i, x_order.shape, y_order.shape, x_store.shape, y_store.shape)
                        finished = False
                        break
                loop_count += 1
                if finished:
                    print(model)
                    loop = False

            print('TEST')
            results[d][fold].append(model.score(x_test_folds, y_test_folds))
            print(results[d][fold])
            fold += 1
            print('DONE')

    # TODO: Decide what to return to make it the same for all models
    # Metrics and model maybe? -> With metrics decide best generalized model.
    # Extract best model metrics
    print(results)
    return results
