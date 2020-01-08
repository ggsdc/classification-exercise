import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from wittgenstein import RIPPER


def rule_induction(x_train, y_train, x_test, y_test):
    final_results = dict()

    df_train = pd.DataFrame(x_train)
    df_train['churn'] = pd.Series(y_train)

    x_test = pd.DataFrame(x_test)
    y_test = pd.DataFrame(y_test)

    model = RIPPER(verbosity=1)
    model.fit(df_train, class_feat="churn")
    y_prediction = model.predict(x_test)
    cm = confusion_matrix(y_test, y_prediction)
    cr = classification_report(y_test, y_prediction, output_dict=True)

    final_results['model'] = model
    final_results['acc'] = (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[1][0] + cm[0][1] + cm[1][1])
    final_results['cm'] = cm
    final_results['cr'] = cr

    print(final_results['acc'])

    return final_results
