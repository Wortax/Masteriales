import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import lightgbm as lgb
from scipy.stats import randint, uniform
import os

def train_and_test(data, labels, name, multiclass=False):
    data = data.dropna()
    labels = labels.loc[data.index]
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=42)

    if multiclass :
        num_classes = labels.nunique()
        lgb_model = lgb.LGBMClassifier(objective='multiclass', num_class=num_classes, random_state=42, verbose=-1)
    else:
        lgb_model = lgb.LGBMClassifier(random_state=42, verbose=-1)

    param_distributions = {
        'n_estimators': randint(50, 300),
        'max_depth': [None] + list(range(10, 50, 10)),
        'min_child_samples': randint(5, 50),
        'num_leaves': randint(20, 150),
        'feature_fraction': uniform(0.5, 0.5),
        'learning_rate': uniform(0.01, 0.3)
    }

    random_search = RandomizedSearchCV(
        estimator=lgb_model,
        param_distributions=param_distributions,
        scoring='accuracy',
        cv=5,
        verbose=1,
        n_jobs=-1,
        n_iter=50,
        random_state=42
    )

    random_search.fit(X_train, y_train)

    best_model = random_search.best_estimator_

    predictions = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    if multiclass:
        probabilities = best_model.predict_proba(X_test)
        f1 = f1_score(y_test, predictions, average='macro')
        auc = roc_auc_score(y_test, probabilities, multi_class='ovr')
    else :
        probabilities = best_model.predict_proba(X_test)[:, 1]
        f1 = f1_score(y_test, predictions)
        auc = roc_auc_score(y_test, probabilities)

    print("LightGBM", name)
    print("Best Parameters:", best_model.get_params())
    print("Accuracy:", round(accuracy, 4))
    print("F1 Score:", round(f1, 4))
    print("AUC:", round(auc, 4))
    print()
    print()
