from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def make_logreg(X_train, X_test, y_train, y_test):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    metrics = {
        'accuracy': accuracy_score(y_test, predictions),
        'precision': precision_score(y_test, predictions),
        'recall': recall_score(y_test, predictions),
        'f1': f1_score(y_test, predictions),
        'AUC': roc_auc_score(y_test, probabilities)
    }
    return metrics


def make_knn(X_train, X_test, y_train, y_test):
    param_grid = {'n_neighbors': range(2, 15)}
    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    model = grid_search.best_estimator_
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    metrics = {
        'accuracy': accuracy_score(y_test, predictions),
        'precision': precision_score(y_test, predictions),
        'recall': recall_score(y_test, predictions),
        'f1': f1_score(y_test, predictions),
        'AUC': roc_auc_score(y_test, probabilities)
    }
    return metrics

def make_models(df, cols, target):
    X = df[cols]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    log_reg_metrics = make_logreg(X_train, X_test, y_train, y_test)
    knn_metrics = make_knn(X_train, X_test, y_train, y_test)
    combined = {'Logistic Regression': log_reg_metrics, 'kNN': knn_metrics}
    return combined