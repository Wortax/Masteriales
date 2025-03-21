import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from catboost import CatBoostClassifier

def train_and_test(data, labels, name, multiclass = False, parameter_tuning_type = "Default"):
    data = data.dropna()
    labels = labels.loc[data.index]
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=42)

    cb_model = CatBoostClassifier(random_state=42, verbose=0,devices='0')

    param_list = {
        'iterations': [50, 100, 150, 200, 300],
        'depth': [4, 6, 8, 10],
        'learning_rate': [0.01, 0.03, 0.1, 0.2],
        'l2_leaf_reg': [1, 3, 5, 7, 9],
        'border_count': [32, 64, 128],
        'bagging_temperature': [0, 0.5, 1, 2, 3]
    }
    if parameter_tuning_type == "Default":
        cb_model.fit(X_train, y_train)
        predictions = cb_model.predict(X_test)
        probabilities = cb_model.predict_proba(X_test)
        best_params = cb_model.get_params()
    else :
        if parameter_tuning_type == "Random":
            search = RandomizedSearchCV(
                estimator=cb_model,
                param_distributions=param_list,
                scoring='f1_weighted' if multiclass else 'accuracy',
                n_iter=20,
                cv=5,
                verbose=1,
                n_jobs=-1,
                random_state=42
            )

        elif parameter_tuning_type == "Grid":
            search = GridSearchCV(
                estimator=cb_model,
                param_grid=param_list,
                scoring='f1_weighted' if multiclass else 'accuracy',
                cv=5,
                verbose=1,
                n_jobs=-1
            )
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        predictions = best_model.predict(X_test)
        probabilities = best_model.predict_proba(X_test)
        best_params = best_model.get_params()


    accuracy = accuracy_score(y_test, predictions)
    if multiclass:
        f1 = f1_score(y_test, predictions, average='weighted')
        auc = roc_auc_score(y_test, probabilities, multi_class='ovr')
    else:
        f1 = f1_score(y_test, predictions)
        auc = roc_auc_score(y_test, probabilities[:, 1])

    results = ""
    results += "CatBoost"+str(name)
    results += "Best Parameters:"+str(best_params)
    results += "Accuracy:"+str(round(accuracy, 4))
    results += "F1 Score:"+str(round(f1, 4))
    results += "AUC:"+str(round(auc, 4))

    return results
