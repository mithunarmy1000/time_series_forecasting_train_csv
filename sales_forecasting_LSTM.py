import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# ------------------ Load Data ------------------
path=r"C:\Users\DELL\Downloads\train.csv\train.csv" #use train.csv file in repository
df=pd.read_csv(path)
df = df[df["store_nbr"] == 1]                  # One store
df = df.groupby("date")["sales"].sum().reset_index()

sales = df["sales"].values.reshape(-1,1)

# ------------------ Normalize ------------------
scaler = MinMaxScaler()
sales = scaler.fit_transform(sales)

# ------------------ Create Sequences ------------------
seq_len = 30
X, y = [], []

for i in range(len(sales)-seq_len):
    X.append(sales[i:i+seq_len])
    y.append(sales[i+seq_len])

X = torch.FloatTensor(np.array(X))
y = torch.FloatTensor(np.array(y))

# Train/Test Split
split = int(len(X)*0.8)

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ------------------ LSTM Model ------------------
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=64,
            batch_first=True
        )
        self.fc = nn.Linear(64,1)

    def forward(self,x):
        out,(h,c)=self.lstm(x)
        return self.fc(h[-1])


model = LSTMModel()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

# ------------------ Training ------------------
epochs = 50

for epoch in range(epochs):

    pred = model(X_train)

    loss = criterion(pred,y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1)%10==0:
        print(f"Epoch {epoch+1}/{epochs} Loss:{loss.item():.6f}")

# ------------------ Prediction ------------------
model.eval()

with torch.no_grad():
    pred = model(X_test)

pred = scaler.inverse_transform(pred.numpy())
actual = scaler.inverse_transform(y_test.numpy())

# ------------------ Plot ------------------
plt.figure(figsize=(10,5))
plt.plot(actual,label="Actual")
plt.plot(pred,label="Predicted")
plt.title("Sales Forecast using LSTM")
plt.xlabel("Days")
plt.ylabel("Sales")
plt.legend()
plt.show()
