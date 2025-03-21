import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def roc_auc_score_multiclass(actual_class, pred_class, average = "macro"):

  unique_class = set(actual_class)
  roc_auc_dict = {}
  for per_class in unique_class:
    other_class = [x for x in unique_class if x != per_class]

    new_actual_class = [0 if x in other_class else 1 for x in actual_class]
    new_pred_class = [0 if x in other_class else 1 for x in pred_class]

    roc_auc = roc_auc_score(new_actual_class, new_pred_class, average = average)
    roc_auc_dict[per_class] = roc_auc

  return roc_auc_dict

def train_and_test(data, labels, name, multiclass = False, parameter_tuning_type = "Default"):
    data = data.dropna()
    labels = labels.loc[data.index]
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=42)

    rf_model = RandomForestClassifier(random_state=42)



    param_list = {
        'max_depth': [3, 5, 7, 10,20,30,40,50],
        'n_estimators': [100, 200, 300, 400, 500],
        'max_features': [10, 20, 30, 40],
        'min_samples_leaf': [1, 2, 4,6,8,10],
        'min_samples_split':[2,6,12,16,20],
    }

    if parameter_tuning_type == "Default" :
        rf_model.fit(X_train, y_train)
        predictions = rf_model.predict(X_test)
        probabilities = rf_model.predict_proba(X_test)[:, 1]
        best_param = rf_model.get_params()
    else:
        if parameter_tuning_type == "Random":
            search = RandomizedSearchCV(
                estimator=rf_model,
                param_distributions=param_list,
                scoring='accuracy',
                cv=5,
                verbose=1,
                n_jobs=-1,
                n_iter=50,
                random_state=42
            )
        elif parameter_tuning_type == "Grid" :
            search = GridSearchCV(estimator=rf_model,
                                  param_grid=param_list,
                                  scoring='accuracy',
                                  cv=5,
                                  verbose=1,
                                  n_jobs=-1)

        search.fit(X_train, y_train)
        best_model = search.best_estimator_

        predictions = best_model.predict(X_test)
        probabilities = best_model.predict_proba(X_test)[:, 1]
        best_param = best_model.get_params()

    accuracy = accuracy_score(y_test, predictions)

    if multiclass:
        f1 = f1_score(y_test, predictions, average='micro')
        auc = roc_auc_score_multiclass(y_test, predictions, average='macro')
        auc = sum(auc.keys()) / len(auc.keys())
    else :
        f1 = f1_score(y_test, predictions)
        auc = roc_auc_score(y_test, probabilities)

    results = ""
    results +="RandomForest "+ str(name)+ " With hyperparameter tuning type: "+str(parameter_tuning_type)+"\n"
    results+="Best Parameters:"+str(best_param)+"\n"
    results+= "Accuracy:"+str(round(accuracy, 4))+"\n"
    results+= "F1 Score:"+str(round(f1, 4))+"\n"
    results+= "AUC:"+str(round(auc, 4))+"\n\n"

    return results



