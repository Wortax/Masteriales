import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

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

X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train.values.squeeze(), dtype=torch.long).to(device)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test.values.squeeze(), dtype=torch.long).to(device)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


class MLPModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MLPModel, self).__init__()
        # Couche 1
        self.fc1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)  # Batch Normalization après la première couche
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)  # Dropout après la première couche

        # Couche 2
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)  # Batch Normalization après la deuxième couche
        self.dropout2 = nn.Dropout(0.2)  # Dropout après la deuxième couche

        # Couche de sortie
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        # Couche 1
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)

        # Couche 2
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)

        # Couche de sortie
        x = self.fc3(x)
        return x


input_size = X_train.shape[1]
num_classes = 6
model = MLPModel(input_size,num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)


epochs = 200
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)  # Déplacer les données sur le GPU
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)  # Applatir les cibles
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}")

model.eval()
all_predictions = []
all_targets = []
all_probabilities = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)  # Déplacer les données sur le GPU
        outputs = model(X_batch)
        predictions = torch.argmax(outputs, dim=1)  # Classe prédite
        # Appliquer softmax pour obtenir les probabilités pour chaque classe
        probabilities = F.softmax(outputs, dim=1)
        all_predictions.extend(predictions.cpu().numpy())
        all_targets.extend(y_batch.cpu().numpy())  # Applatir les cibles
        all_probabilities.extend(probabilities.cpu().numpy())

all_predictions = np.array(all_predictions)
all_targets = np.array(all_targets)
all_probabilities = np.array(all_probabilities)

accuracy = accuracy_score(all_targets, all_predictions)
f1 = f1_score(all_targets, all_predictions, average='macro')
auc = roc_auc_score(all_targets, all_probabilities, multi_class='ovr')

print("Classification Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"Area Under the Curve (AUC): {auc:.4f}")
