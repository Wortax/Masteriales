import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import RandomizedSearchCV

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

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

class ResNetBlock(nn.Module):
    def __init__(self, input_size, dropout=0.3):
        super(ResNetBlock, self).__init__()
        self.bn1 = nn.BatchNorm1d(input_size)
        self.fc1 = nn.Linear(input_size, input_size)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(input_size, input_size)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.dropout2(out)
        out += residual
        return out


class Prediction(nn.Module):
    def __init__(self, input_size,num_classes):
        super(Prediction, self).__init__()
        self.bn = nn.BatchNorm1d(input_size)
        self.fc = nn.Linear(input_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        return self.fc(x)


class ResNetTabular(nn.Module):
    def __init__(self, input_size,num_classes, hidden_size=128, num_blocks=3, dropout=0.3):
        super(ResNetTabular, self).__init__()
        self.fc_in = nn.Linear(input_size, hidden_size)

        self.resnet_blocks = nn.ModuleList([ResNetBlock(hidden_size, dropout=dropout) for _ in range(num_blocks)])

        self.prediction = Prediction(hidden_size,num_classes)

    def forward(self, x):
        x = self.fc_in(x)
        for block in self.resnet_blocks:
            x = block(x)
        return self.prediction(x)

class TorchResNetModel(BaseEstimator, ClassifierMixin):
    def __init__(self, input_size,num_classes, hidden_size=128, num_blocks=3, dropout=0.3, learning_rate=0.001, weight_decay=1e-4, epochs=10):
        self.input_size = input_size
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_blocks = num_blocks
        self.dropout = dropout
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
        self.model = ResNetTabular(self.input_size,self.num_classes, self.hidden_size, self.num_blocks, self.dropout).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            for X_batch, y_batch in dataloader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
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
    'hidden_size': [64, 128, 256],
    'num_blocks': [1, 2, 3, 4],
    'dropout': [0.2, 0.3, 0.4],
    'learning_rate': [0.001, 0.01, 0.0001],
    'weight_decay': [1e-4, 1e-5, 1e-6],
    'epochs': [10, 20, 30]
}

model = TorchResNetModel(input_size=X_train.shape[1],num_classes=6)
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_grid,
    n_iter=10,
    scoring='accuracy',
    cv=3,
    verbose=1,
    random_state=42
)
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
