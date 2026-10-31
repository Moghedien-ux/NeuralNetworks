import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

distances = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype = torch.float32).to(device)
times = torch.tensor([[6.69], [12.11], [16.77], [22.21]], dtype = torch.float32).to(device)

model = nn.Sequential(
    nn.Linear(1, 3),
    nn.ReLU(),
    nn.Linear(3, 1)
).to(device)

loss_function = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr = 0.01)

for epoch in range(500):
    optimizer.zero_grad()
    outputs = model(distances)
    loss = loss_function(outputs, times)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch + 1}: Loss = {loss.item()}")


with torch.no_grad():
    tes_distance = torch.tensor([[25.0]], dtype = torch.float32).to(device)
    predicted_time = model(tes_distance)
    print(f"Predicted for 25: {predicted_time.item():.1f} minutes")

layer = model[0]
weights = layer.weight.data.cpu().numpy()
bias = layer.bias.data.cpu().numpy()
print(weights)
print(bias)