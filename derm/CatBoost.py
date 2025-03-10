import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from catboost import CatBoostClassifier

def train_and_test(data, labels, name, multiclass = False):
    data = data.dropna()
    labels = labels.loc[data.index]
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=42)

    cb_model = CatBoostClassifier(random_state=42, verbose=0,devices='0')

    param_distributions = {
        'iterations': [50, 100, 150, 200, 300],
        'depth': [4, 6, 8, 10],
        'learning_rate': [0.01, 0.03, 0.1, 0.2],
        'l2_leaf_reg': [1, 3, 5, 7, 9],
        'border_count': [32, 64, 128],
        'bagging_temperature': [0, 0.5, 1, 2, 3]
    }

    if multiclass:
        random_search = RandomizedSearchCV(
            estimator=cb_model,
            param_distributions=param_distributions,
            scoring='f1_weighted',
            n_iter=20,
            cv=5,
            verbose=1,
            n_jobs=-1,
            random_state=42
        )
    else:
        random_search = RandomizedSearchCV(
            estimator=cb_model,
            param_distributions=param_distributions,
            scoring='accuracy',
            n_iter=20,
            cv=5,
            verbose=1,
            n_jobs=-1,
            random_state=42
        )
    random_search.fit(X_train, y_train)

    best_model = random_search.best_estimator_

    predictions = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    if multiclass:
        probabilities = best_model.predict_proba(X_test)
        f1 = f1_score(y_test, predictions, average='weighted')
        auc = roc_auc_score(y_test, probabilities, multi_class='ovr')
    else:
        probabilities = best_model.predict_proba(X_test)[:, 1]
        f1 = f1_score(y_test, predictions)
        auc = roc_auc_score(y_test, probabilities)

    print("CatBoost", name)
    print("Best Parameters:", best_model.get_params())
    print("Accuracy:", round(accuracy, 4))
    print("F1 Score:", round(f1, 4))
    print("AUC:", round(auc, 4))
    print()
    print()
