import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import RandomizedSearchCV

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_path = "derm.csv"
x = pd.read_csv(data_path)

label_pass = "labels_derm.csv"
y = pd.read_csv(label_pass)
y["classe"] = y["classe"] - 1

for col in x.columns:
    if x[col].dtype == 'object':
        le = LabelEncoder()
        x[col] = le.fit_transform(x[col])

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)  # 80% train

class MLPModel(nn.Module):
    def __init__(self, input_size, num_classes, hidden_layer1, hidden_layer2, dropout1, dropout2):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_layer1)
        self.bn1 = nn.BatchNorm1d(hidden_layer1)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout1)

        self.fc2 = nn.Linear(hidden_layer1, hidden_layer2)
        self.bn2 = nn.BatchNorm1d(hidden_layer2)
        self.dropout2 = nn.Dropout(dropout2)

        self.fc3 = nn.Linear(hidden_layer2, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        return x

class TorchModel(BaseEstimator, ClassifierMixin):
    def __init__(self, input_size, num_classes, hidden_layer1=128, hidden_layer2=64, dropout1=0.3, dropout2=0.2, learning_rate=0.001, weight_decay=1e-4, epochs=10):
        self.input_size = input_size
        self.num_classes = num_classes
        self.hidden_layer1 = hidden_layer1
        self.hidden_layer2 = hidden_layer2
        self.dropout1 = dropout1
        self.dropout2 = dropout2
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.classes_ = None
        self.model = None

    def fit(self, X, y):
        X_tensor = torch.tensor(X.values, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y.values.squeeze(), dtype=torch.long).to(device)

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

        self.classes_ = np.unique(y)
        self.model = MLPModel(self.input_size, self.num_classes, self.hidden_layer1, self.hidden_layer2, self.dropout1, self.dropout2).to(
            device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            for X_batch, y_batch in dataloader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)  # Déplacer les données sur le GPU
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {running_loss / len(dataloader):.4f}")

    def predict(self, X):
        X_tensor = torch.tensor(X.values, dtype=torch.float32).to(device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor).squeeze()
            return torch.argmax(outputs, dim=1)  # Classe prédite

    def predict_proba(self, X):
        X_tensor = torch.tensor(X.values, dtype=torch.float32).to(device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor).squeeze()
            return F.softmax(outputs, dim=1).cpu().numpy()

param_grid = {
    'hidden_layer1': [64, 128, 256],
    'hidden_layer2': [32, 64, 128],
    'dropout1': [0.2, 0.3, 0.4],
    'dropout2': [0.2, 0.3, 0.4],
    'learning_rate': [0.001, 0.01, 0.0001],
    'weight_decay': [1e-4, 1e-5, 1e-6],
    'epochs': [10, 20, 30, 50]
}

model = TorchModel(input_size=X_train.shape[1], num_classes=6)
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=10, scoring='accuracy', cv=3, verbose=1, random_state=42)
random_search.fit(X_train, y_train)

print("Best Parameters:", random_search.best_params_)
print("Best Score:", random_search.best_score_)

best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='macro')
roc_auc = roc_auc_score(y_test, best_model.predict_proba(X_test), multi_class='ovr')

print("Test Accuracy:", accuracy)
print("F1 Score:", f1)
print("ROC AUC Score:", roc_auc)
