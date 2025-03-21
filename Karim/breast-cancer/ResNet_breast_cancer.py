import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset


data_path = "breast-cancer.csv"
x = pd.read_csv(data_path)

label_pass = "labels_breast-cancer.csv"
y = pd.read_csv(label_pass)


for col in x.columns:
    if x[col].dtype == 'object':
        le = LabelEncoder()
        x[col] = le.fit_transform(x[col])


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)  # 80% train


X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)


train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


class ResNetBlock(nn.Module):
    def __init__(self, input_size):
        super(ResNetBlock, self).__init__()
        self.bn1 = nn.BatchNorm1d(input_size)
        self.fc1 = nn.Linear(input_size, input_size)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(input_size, input_size)
        self.dropout2 = nn.Dropout(0.3)

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
    def __init__(self, input_size):
        super(Prediction, self).__init__()
        self.bn = nn.BatchNorm1d(input_size)
        self.fc = nn.Linear(input_size, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        return self.fc(x)


class ResNetTabular(nn.Module):
    def __init__(self, input_size, num_blocks=3):
        super(ResNetTabular, self).__init__()
        self.fc_in = nn.Linear(input_size, 128)

        self.resnet_blocks = nn.ModuleList([ResNetBlock(128) for _ in range(num_blocks)])

        self.prediction = Prediction(128)

    def forward(self, x):
        x = self.fc_in(x)
        for block in self.resnet_blocks:
            x = block(x)
        return self.prediction(x)



input_size = X_train.shape[1]
model = ResNetTabular(input_size)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


epochs = 200
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch).squeeze()
        loss = criterion(outputs, y_batch.squeeze())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")


model.eval()
all_predictions = []
all_targets = []
all_probabilities = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch).squeeze()
        predictions = (outputs >= 0.5).float()
        all_predictions.extend(predictions.cpu().numpy())
        all_targets.extend(y_batch.squeeze().cpu().numpy())
        all_probabilities.extend(outputs.cpu().numpy())

all_predictions = np.array(all_predictions)
all_targets = np.array(all_targets)
all_probabilities = np.array(all_probabilities)

accuracy = accuracy_score(all_targets, all_predictions)
f1 = f1_score(all_targets, all_predictions)
auc = roc_auc_score(all_targets, all_probabilities)

print("\nClassification Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"Area Under the Curve (AUC): {auc:.4f}")

