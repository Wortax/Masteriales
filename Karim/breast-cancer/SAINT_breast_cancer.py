import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.base import BaseEstimator, ClassifierMixin
from saint.models.model import TabAttention
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_path = "breast-cancer.csv"
x = pd.read_csv(data_path)

label_path = "labels_breast-cancer.csv"
y = pd.read_csv(label_path)


for col in x.columns:
    if x[col].dtype == 'object':
        le = LabelEncoder()
        x[col] = le.fit_transform(x[col])



def split_data(X, Y):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test

X_train, X_val, X_test, y_train, y_val, y_test = split_data(x, y)


X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).to(device)
X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long).to(device)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.long).to(device)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long).to(device)

class TabAttentionWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, dim=32, depth=6, heads=8, dim_head=16, dim_out=1, mlp_hidden_mults=(4, 2),
                 mlp_act=torch.nn.ReLU(), attentiontype='col', attn_dropout=0.1, ff_dropout=0.1,
                 lastmlp_dropout=0.1, learning_rate=1e-3, batch_size=64, epochs=50, device=device):
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.dim_head = dim_head
        self.dim_out = dim_out
        self.mlp_hidden_mults = mlp_hidden_mults
        self.mlp_act = mlp_act
        self.attentiontype = attentiontype
        self.attn_dropout = attn_dropout
        self.ff_dropout = ff_dropout
        self.lastmlp_dropout = lastmlp_dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device
        self.model = None

    def fit(self, X, y):
        num_features = X.shape[1]
        self.model = TabAttention(
            categories=[],  # Aucune variable catégorique ici
            num_continuous=num_features,
            dim=self.dim,
            depth=self.depth,
            heads=self.heads,
            dim_head=self.dim_head,
            dim_out=self.dim_out,
            mlp_hidden_mults=self.mlp_hidden_mults,
            mlp_act=self.mlp_act,
            attentiontype=self.attentiontype,
            attn_dropout=self.attn_dropout,
            ff_dropout=self.ff_dropout,
            lastmlp_dropout=self.lastmlp_dropout
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = torch.nn.CrossEntropyLoss()

        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            outputs = self.model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

        return self

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X)
            return torch.argmax(outputs, dim=1).cpu().numpy()

    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X)
            return torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()


param_grid = {
    'dim': [32, 64],
    'depth': [4, 6],
    'heads': [4, 8],
    'dim_head': [16, 32],
    'learning_rate': [1e-4, 1e-3],
    'batch_size': [32, 64],
    'epochs': [30, 50]
}


model = TabAttentionWrapper()
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_grid,
    n_iter=10,
    scoring='accuracy',
    cv=3,
    verbose=2,
    random_state=42,
)


random_search.fit(X_train_tensor, y_train_tensor)


print("Best Parameters:", random_search.best_params_)
print("Best Score:", random_search.best_score_)

best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test_tensor)
y_proba = best_model.predict_proba(X_test_tensor)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba[:, 1])

print("Test Accuracy:", accuracy)
print("F1 Score:", f1)
print("ROC AUC Score:", roc_auc)
