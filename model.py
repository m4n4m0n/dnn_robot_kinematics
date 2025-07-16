## Martina Kraußer, 2025

import torch
import torch.nn as nn
import torch.nn.functional as F 



class DNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.out = nn.Linear(hidden_sizes[2], output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.out(x)

    def train_model(self, X_train, y_train, X_val, y_val, optimizer, criterion, epochs=100):
        history = {'train_loss': [], 'val_loss': []}
        print("Training started...")
        for epoch in range(epochs):
            self.train()
            optimizer.zero_grad()
            output = self(X_train)
            loss = criterion(output, y_train)
            loss.backward()
            optimizer.step()

            self.eval()
            with torch.no_grad():
                val_output = self(X_val)
                val_loss = criterion(val_output, y_val)

            history['train_loss'].append(loss.item())
            history['val_loss'].append(val_loss.item())
        print("Training finished. ")
        return history, val_loss.item()

    def test_model(self, X_test, y_test, criterion):
        print("Testing started...")
        self.eval()
        with torch.no_grad():
            prediction = self(X_test)
            loss = criterion(prediction, y_test)

        print("Testing finished. ")
        return loss



if __name__ == '__main__':
    model = DNN(input_size=7, hidden_sizes=[128,256,128], output_size=7)
    print(model)

    # Test with random input
    random_input = torch.randn(1, 7)  # 1 Sample, 7 Joint Angles
    output = model(random_input)

    # Print input and output for verification
    print("Input (Joint Angles):", random_input)
    print("Output (Translation + Quaternion):", output)