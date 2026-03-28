import torch
import torch.nn as nn
import torch.optim as optim
import os

# Corrected Paths (Relative to backend folder)
os.makedirs('models/deepfake', exist_ok=True)

class DeepfakeCNN(nn.Module):
    def __init__(self):
        super(DeepfakeCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 32 * 32)
        x = torch.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

def train_demo():
    model = DeepfakeCNN()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    X_train = torch.randn(10, 3, 128, 128)
    y_train = torch.randint(0, 2, (10, 1)).float()

    print("Training Vision Forensic Demo model...")
    for epoch in range(5):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        print(f"Vision Training Progress: {((epoch+1)/5)*100}%")

    torch.save(model.state_dict(), 'models/deepfake/model.pth')
    print("✅ Vision Forensic Model Saved!")

if __name__ == '__main__':
    train_demo()
