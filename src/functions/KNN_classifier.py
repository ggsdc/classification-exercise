import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier


# TODO: check the distance to be used in class slides
# TODO: try with different weights and distances to get different results -> Brute force
# TODO: brute force number of neighbors.

def knn_classifier(X_train, y_train, X_test, y_test, folds, n_attributes):
    distances = [1, 2]  # Manhattan and euclidean
    results = dict()

    for d in distances:
        results[d] = dict()
        fold = 1
        for idx_train, idx_test in folds.split(X_train, y_train):
            results[d][fold] = list()
            X_train_folds = X_train[idx_train]
            X_test_folds = X_train[idx_test]
            y_train_folds = y_train[idx_train]
            y_test_folds = y_train[idx_test]
            loop = True
            # HERE WE APPLY THE LOGIC FOR THE CNN
            # WE ARE GOING TO MODIFY THE LOGIC SLIGHTLY.
            # WE START WITH THE FIRST SAMPLE DIRECTLY.
            order_idx = np.delete(np.arange(X_train_folds.shape[0]), 0)
            X_store = X_train_folds[0]
            y_store = y_train_folds[0]
            X_order = X_train_folds[order_idx]
            y_order = y_train_folds[order_idx]
            print(X_order.shape, y_order.shape, X_store.shape, y_store.shape)
            print(y_store)
            loop_count = 1

            while loop:
                model = KNeighborsClassifier(n_neighbors=1, weights='uniform', algorithm='kd_tree', p=d)
                model.fit(X_store.reshape(loop_count, n_attributes), y_store.reshape(loop_count, 1))
                finished = True
                for i in range(X_order.shape[0]):
                    pred = model.predict(X_order[i].reshape(1, n_attributes))
                    if pred != y_order[i]:
                        X_store = np.concatenate(
                            (X_store.reshape(loop_count, n_attributes), X_order[i].reshape(1, n_attributes)))
                        y_store = np.concatenate((y_store.reshape(loop_count, 1), y_order[i].reshape(1, 1)))

                        order_idx = np.delete(order_idx, i)

                        X_order = X_train_folds[order_idx]
                        y_order = y_train_folds[order_idx]

                        print(i, X_order.shape, y_order.shape, X_store.shape, y_store.shape)
                        finished = False
                        break
                loop_count += 1
                if finished:
                    print(model)
                    loop = False

            print('TEST')
            results[d][fold].append(model.score(X_test_folds, y_test_folds))
            print(results[d][fold])
            fold += 1
            print('DONE')




    # TODO: Decide what to return to make it the same for all models
    # Metrics and model maybe? -> With metrics decide best generalized model.
    # Extract best model metrics
    print(results)
    return results