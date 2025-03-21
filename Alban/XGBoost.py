import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from scipy.stats import randint, uniform


def train_and_test(data, labels, name, multiclass = False, parameter_tuning_type = "Default"):
    data = data.dropna()
    labels = labels.loc[data.index]

    unique_classes = sorted(labels.unique())
    class_mapping = {old: new for new, old in enumerate(unique_classes)}
    labels = labels.map(class_mapping)

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=42)

    if multiclass :
        xgb_model = XGBClassifier(random_state=42, objective = 'multi:softmax', eval_metric = 'mlogloss')
    else :
        xgb_model = XGBClassifier(random_state=42, eval_metric='logloss', objective = 'binary:logistic')


    param_list = {
        'n_estimators': randint(50, 300),
        'max_depth': randint(10, 40),
        'learning_rate': uniform(0.01, 0.3),
        'colsample_bytree': uniform(0.5, 0.5),
        'subsample': uniform(0.5, 0.5)
    }

    if parameter_tuning_type == "Default" :
        xgb_model.fit(X_train, y_train)
        best_param = xgb_model.get_xgb_params()
        predictions = xgb_model.predict(X_test)
        probabilities = xgb_model.predict_proba(X_test)
    else:
        if parameter_tuning_type == "Random":
            search = RandomizedSearchCV(
                estimator=xgb_model,
                param_distributions=param_list,
                n_iter=20,
                scoring='accuracy',
                cv=5,
                verbose=1,
                n_jobs=-1,
                random_state=42
            )
        elif parameter_tuning_type == "Grid" :
            search = GridSearchCV(estimator=xgb_model,
                                  param_grid=param_list,
                                  scoring='accuracy',
                                  cv=5,
                                  verbose=1,
                                  n_jobs=-1)
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        predictions = best_model.predict(X_test)
        best_param = best_model.get_xgb_params()
        probabilities = best_model.predict_proba(X_test)





    accuracy = accuracy_score(y_test, predictions)

    if multiclass:
        f1 = f1_score(y_test, predictions, average='weighted')
        auc = roc_auc_score(y_test, probabilities, multi_class="ovr", average="macro")
    else:
        f1 = f1_score(y_test, predictions)
        auc = roc_auc_score(y_test, probabilities[:, 1])

    results = ""
    results += "XGBoost"+str(name) + "\n"
    results += "Best parameters:"+str(best_param)+"\n"
    results += "Accuracy:"+str(round(accuracy, 4))+"\n"
    results += "F1 Score:"+str(round(f1, 4)) + "\n"
    results += "AUC:"+str(round(auc, 4))+"\n\n"

    return results


