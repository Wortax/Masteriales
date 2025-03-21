import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import RandomizedSearchCV

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_path = "diabetes.csv"
x = pd.read_csv(data_path)

label_pass = "labels_diabetes.csv"
y = pd.read_csv(label_pass)

for col in x.columns:
    if x[col].dtype == 'object':
        le = LabelEncoder()
        x[col] = le.fit_transform(x[col])


scaler = StandardScaler()
x = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)

def split_data(X, Y):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    return X_train.values, X_val.values, X_test.values, y_train.values.ravel(), y_val.values.ravel(), y_test.values.ravel()

X_train, X_val, X_test, y_train, y_val, y_test = split_data(x, y)

param_grid = {
    'n_d': [8, 16, 32],
    'n_a': [8, 16, 32],
    'n_steps': [3, 5, 7],
    'gamma': [1.0, 1.5, 2.0],
    'lambda_sparse': [0.0001, 0.001, 0.01],
    'momentum': [0.02, 0.03, 0.04],
    'batch_size': [8, 16, 32, 64],
    'virtual_batch_size': [2, 4],
    'max_epochs': [50, 100],
    'cat_idxs': [[]],
    'cat_dims': [[]],
    'cat_emb_dim': [1],
    'n_independent': [1, 2],
    'n_shared': [1, 2],
    'mask_type': ['sparsemax', 'entmax'],
}

class TabNetWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, n_d=8, n_a=8, n_steps=3, gamma=1.3, cat_idxs=[], cat_dims=[], cat_emb_dim=1,
                 n_shared=2, n_independent=2, lambda_sparse=0.0001, momentum=0.02,
                 batch_size=1024, virtual_batch_size=128, max_epochs=100, mask_type="sparsemax", **kwargs):

        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.cat_idxs = cat_idxs
        self.cat_dims = cat_dims
        self.cat_emb_dim = cat_emb_dim
        self.n_shared = n_shared
        self.n_independent = n_independent
        self.lambda_sparse = lambda_sparse
        self.momentum = momentum
        self.batch_size = batch_size
        self.virtual_batch_size = virtual_batch_size
        self.max_epochs = max_epochs
        self.mask_type = mask_type
        self.kwargs = kwargs
        self.classes_ = None
        self.model = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.model = TabNetClassifier(
            n_d=self.n_d,
            n_a=self.n_a,
            n_steps=self.n_steps,
            gamma=self.gamma,
            cat_idxs=self.cat_idxs,
            cat_dims=self.cat_dims,
            cat_emb_dim=self.cat_emb_dim,
            n_shared=self.n_shared,
            n_independent=self.n_independent,
            lambda_sparse=self.lambda_sparse,
            momentum=self.momentum,
            mask_type=self.mask_type,
            device_name=device.type,
            **self.kwargs
        )

        self.model.fit(
            X_train=X, y_train=y,
            eval_set=[(X_val, y_val)],
            eval_name=["val"],
            eval_metric=["accuracy"],
            max_epochs=self.max_epochs,
            patience=10,
            batch_size=self.batch_size,
            virtual_batch_size=self.virtual_batch_size
        )
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

model = TabNetWrapper()

random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_grid,
    n_iter=10,
    scoring='accuracy',
    cv=3,
    verbose=2,
    random_state=42,
)

random_search.fit(X_train, y_train)

print("Best Parameters:", random_search.best_params_)
print("Best Score:", random_search.best_score_)

best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba[:, 1])

print("Test Accuracy:", accuracy)
print("F1 Score:", f1)
print("ROC AUC Score:", roc_auc)
