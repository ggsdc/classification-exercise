from sklearn.neighbors import KNeighborsClassifier

# TODO: check the distance to be used in class slides
# TODO: try with different weights and distances to get different results -> Brute force
# TODO: brute force number of neighbors.

def knn_classifier(X_train, y_train, X_test, y_test):
    weights = ['uniform', 'distance']
    neighbors = [3, 5, 7, 9, 11]
    distances = [1, 2] # Manhattan and euclidean
    best_model = KNeighborsClassifier()
    best_accuracy = 0
    for n in neighbors:
        for w in weights:
            for d in distances:
                model = KNeighborsClassifier(n_neighbors=n, weights=w, algorithm='auto', p=d)
                model.fit(X_train, y_train)
                y_predict = model.predict(X_test)
                y_predict_prob = model.predict(X_test)
                # TODO: calcuate accuracy and compare with best
                # Maybe it does not have to be accuracy but it can be any other metric.
                accuracy = 100
                if accuracy > best_accuracy:
                    best_model = model


    # TODO: Decide what to return to make it the same for all models
    # Metrics and model maybe? -> With metrics decide best generalized model.
    # Extract best model metrics
    return True