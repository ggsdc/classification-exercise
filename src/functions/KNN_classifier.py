from sklearn.neighbors import KNeighborsClassifier

# TODO: check the distance to be used in class slides
# TODO: try with different weights and distances to get different results -> Brute force
# TODO: brute force number of neighbors.

def knn_classifier(X_train, y_train, X_test, y_test, folds):
    distances = [1, 2] # Manhattan and euclidean
    best_model = KNeighborsClassifier()
    best_accuracy = 0
    results = dict()
    for d in distances:
        results[d] = list()
        print(d)
        count = 1
        for idx_train, idx_test in folds.split(X_train, y_train):
            print(count)
            X_train_folds = X_train[idx_train]
            X_test_folds = X_train[idx_test]
            y_train_folds = y_train[idx_train]
            y_test_folds = y_train[idx_test]
            loop = True
            # HERE WE APPLY THE LOGIC FOR THE CNN
            # FIRST WE COMPUTE ALL RANKS AND DISTANCES

            # WE HAVE TO LOOP
            while loop == True:


                break

            model = KNeighborsClassifier(n_neighbors=1, weights='uniform', algorithm='kd_tree', p=d)
            # print(model)
            model.fit(X_train_folds, y_train_folds.reshape(X_train_folds.shape[0],))
            print('TEST')
            results[d].append(model.score(X_test_folds, y_test_folds))
            print('DONE')
            count += 1




    # TODO: Decide what to return to make it the same for all models
    # Metrics and model maybe? -> With metrics decide best generalized model.
    # Extract best model metrics
    return results