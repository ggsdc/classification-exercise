import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier


# TODO: check the distance to be used in class slides
# TODO: try with different weights and distances to get different results -> Brute force
# TODO: brute force number of neighbors.

def knn_classifier(x_train, y_train, x_test, y_test, folds, n_attributes):
    print("CNN")
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

                        x_order = x_train_folds[order_idx]
                        y_order = y_train_folds[order_idx]

                        finished = False
                        break

                if loop_count % 100 == 0:
                    print(loop_count, 'iterations')
                    print((d, fold, i), x_order.shape, y_order.shape, x_store.shape, y_store.shape)

                loop_count += 1
                if finished:
                    print('Finished fold', fold)
                    print(model, x_order.shape, y_order.shape, x_store.shape, y_store.shape)
                    loop = False

            results[d][fold] = model.score(x_test_folds, y_test_folds)
            print(d, fold, results[d][fold])
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
    order_idx = np.delete(np.arange(x_train.shape[0]), 0)
    x_store = x_train[0]
    y_store = y_train[0]
    x_order = x_train[order_idx]
    y_order = y_train[order_idx]
    loop = True
    loop_count = 1

    best_model = KNeighborsClassifier(n_neighbors=1, weights='uniform', algorithm='kd_tree', p=best_config)
    while loop:
        best_model = KNeighborsClassifier(n_neighbors=1, weights='uniform', algorithm='kd_tree', p=best_config)
        best_model.fit(x_store.reshape(loop_count, n_attributes), y_store.reshape(loop_count, 1))
        finished = True
        for i in range(x_order.shape[0]):
            predicted = best_model.predict(x_order[i].reshape(1, n_attributes))
            if predicted != y_order[i]:
                x_store = np.concatenate(
                    (x_store.reshape(loop_count, n_attributes), x_order[i].reshape(1, n_attributes)))
                y_store = np.concatenate((y_store.reshape(loop_count, 1), y_order[i].reshape(1, 1)))

                order_idx = np.delete(order_idx, i)

                x_order = x_train[order_idx]
                y_order = y_train[order_idx]

                finished = False
                break

        loop_count += 1
        if finished:
            print(best_model, x_order.shape, y_order.shape, x_store.shape, y_store.shape)
            loop = False

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
