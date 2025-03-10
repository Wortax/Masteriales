import torch
from hyperfast import HyperFastClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV



def train_and_test(data, labels, name):
    data = data.dropna()
    labels = labels.loc[data.index]
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=42)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    hypfast_model = HyperFastClassifier(device=device)

    param_distributions = {
        'n_ensemble': [1, 4, 8, 16, 32],
        'batch_size': [1024, 2048],
        'nn_bias': [True, False],
        'stratify_sampling': [True, False],
        'optimization': [None, 'optimize', 'ensemble_optimize'],
        'optimize_steps': [1, 4, 8, 16, 32, 64, 128],
        'seed': list(range(10))
    }

    #search = RandomizedSearchCV(hypfast_model, param_distributions)
    search = GridSearchCV(hypfast_model, param_distributions, cv=3, scoring='accuracy', verbose=1)



    search.fit(X_train, y_train)
    print("fitting model")

    best_model = search.best_estimator_
    print("predicting")
    predictions = best_model.predict(X_test)
    probabilities = best_model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    auc = roc_auc_score(y_test, probabilities)

    print("HyperFast", name)
    print("Best Parameters:", best_model.get_params())
    print("Accuracy:", round(accuracy, 4))
    print("F1 Score:", round(f1, 4))
    print("AUC:", round(auc, 4))
    print()


