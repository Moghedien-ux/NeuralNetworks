import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Set seed for reproducibility
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

# Features: [Distance (miles), Package Weight (kg), Traffic Congestion Index (1 to 5)]
num_samples = 150
distances = torch.rand(num_samples, 1) * 15.0 + 1.0  # 1 to 16 miles
weights = torch.rand(num_samples, 1) * 10.0 + 0.5  # 0.5 to 10.5 kg
traffic = torch.randint(1, 6, (num_samples, 1)).float()  # Traffic level 1 to 5

# Combine features into a single matrix X (shape: [150, 3])
X = torch.cat([distances, weights, traffic], dim=1)

# Realistic target generation: Time depends on distance, weight, traffic + random noise
times = 6.0 + (X[:, 0:1] * 3.2) + (X[:, 1:2] * 1.5) + (X[:, 2:3] * 4.5) + torch.randn(num_samples, 1) * 1.8

train_size = int(0.8 * num_samples)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = times[:train_size], times[train_size:]

# Compute mean and std strictly from training data to prevent data leakage
X_mean, X_std = X_train.mean(dim=0), X_train.std(dim=0)
y_mean, y_std = y_train.mean(), y_train.std()

X_train_norm = (X_train - X_mean) / X_std
X_test_norm = (X_test - X_mean) / X_std
y_train_norm = (y_train - y_mean) / y_std
y_test_norm = (y_test - y_mean) / y_std

# Wrap into DataLoader for mini-batch training (Industry Standard)
train_dataset = TensorDataset(X_train_norm, y_train_norm)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)


class DeliveryPredictorNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),  # Normalizes activations for training stability
            nn.ReLU(),
            nn.Dropout(0.1),  # Prevents overfitting by randomly dropping neurons
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)  # Output layer predicting normalized time
        )

    def forward(self, x):
        return self.network(x)


model = DeliveryPredictorNN(input_dim=3).to(device)

loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=40, factor=0.5)

epochs = 400
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()
        predictions = model(batch_x)
        loss = loss_function(predictions, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    # Evaluate validation loss periodically
    if (epoch + 1) % 50 == 0:
        model.eval()
        with torch.no_grad():
            test_preds = model(X_test_norm.to(device))
            test_loss = loss_function(test_preds, y_test_norm.to(device))
        scheduler.step(test_loss)
        print(
            f"Epoch {epoch + 1:3d} | Train Loss: {epoch_loss / len(train_loader):.4f} | Test Loss: {test_loss.item():.4f}")

print("\n--- Training Complete ---")

model.eval()
with torch.no_grad():
    # New delivery scenario: Distance = 8.5 miles, Weight = 4.0 kg, Traffic Index = 4 (heavy)
    new_case = torch.tensor([[8.5, 4.0, 4.0]], dtype=torch.float32).to(device)
    new_case_norm = (new_case - X_mean.to(device)) / X_std.to(device)

    pred_norm = model(new_case_norm)
    pred_actual = (pred_norm * y_std.to(device)) + y_mean.to(device)

    final_minutes = pred_actual.item()
    print(f"Scenario Specs -> Distance: 8.5 mi | Weight: 4.0 kg | Traffic Level: 4/5")
    print(f"Estimated Delivery Time: {final_minutes:.1f} minutes")

    # Advanced Business Logic
    if final_minutes > 60:
        print("Decision: High Delay Risk! Assign Express Courier + Notify Customer.")
    elif final_minutes > 40:
        print("Decision: Standard Car Delivery. Monitor live navigation.")
    else:
        print("Decision: Fast Delivery. Eligible for Eco-Bike Dispatch.")