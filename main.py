## Martina Kraußer, 2025

import os
import pickle
import torch
import zipfile

import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim

from model import DNN
from dataloader import load_csv
from sklearn.model_selection import train_test_split


class train_test_loop():
    def __init__(self, mode='FK', lr=0.01, optimizer_type='SGD', loss_type='MSE'):
        self.mode = mode  # 'FK' for forward kinematik or 'IK' for inverse kinematik

        #Load Data
        with open('data.pkl', 'rb') as f:
            data = pickle.load(f)

        if self.mode == 'FK':
            X = data['q_solution']
            y = np.hstack((data['translation'], data['quaternion']))
        elif self.mode == 'IK':
            X = np.hstack((data['translation'], data['quaternion']))
            y = data['q_solution']
        else:
            raise ValueError("Mode must be either 'FK' or 'IK'")

        # Split Data
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=2/9, random_state=42)

        # Torch-Konvertierung
        self.X_train = torch.tensor(X_train, dtype=torch.float32)
        self.y_train = torch.tensor(y_train, dtype=torch.float32)
        self.X_val = torch.tensor(X_val, dtype=torch.float32)
        self.y_val = torch.tensor(y_val, dtype=torch.float32)
        self.X_test = torch.tensor(X_test, dtype=torch.float32)
        self.y_test = torch.tensor(y_test, dtype=torch.float32)

        # Initialize
        input_size = X.shape[1]
        output_size = y.shape[1]
        self.net = DNN(input_size=input_size, hidden_sizes=[128, 256, 128], output_size=output_size)

        # Loss Function
        if loss_type == 'MSE':
            self.criterion = nn.MSELoss()
        elif loss_type == 'MAE':
            self.criterion = nn.L1Loss()
        else:
            raise ValueError("Loss type must be 'MSE' or 'MAE'")

        #Optimizer
        if optimizer_type == 'SGD':
            self.optimizer = optim.SGD(self.net.parameters(), lr=lr, weight_decay=1e-4)
        elif optimizer_type == 'Adam':
            self.optimizer = optim.Adam(self.net.parameters(), lr=lr, weight_decay=1e-4)
        elif optimizer_type == 'RMSprop':
            self.optimizer = optim.RMSprop(self.net.parameters(), lr=lr, weight_decay=1e-4)
        else:
            raise ValueError("Unknown optimizer")

    def train(self, epochs=100):
        print("Training started...")
        for epoch in range(epochs):
            self.net.train()
            output = self.net(self.X_train)
            loss = self.criterion(output, self.y_train)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.net.eval()
            with torch.no_grad():
                val_output = self.net(self.X_val)
                val_loss = self.criterion(val_output, self.y_val)

            print(f"[{epoch+1:03d}] Train Loss: {loss.item():.6f} | Val Loss: {val_loss.item():.6f}")

    def test(self):
        print("Testing started...")
        self.net.eval()
        with torch.no_grad():
            prediction = self.net(self.X_test)
            loss = self.criterion(prediction, self.y_test)
        print(f"Test Loss: {loss.item():.6f}")



if __name__=='__main__':
    # Load Data
    position, quaternion, q_solution = load_csv()

    X = np.hstack((np.array(position).reshape(len(position), -1) ,
                   np.array(quaternion).reshape(len(quaternion), -1)))  # Input
    y = np.array(q_solution).reshape(len(q_solution), -1)  # Output

    print(X[0])
    print(y[0])

    # Split: 70:20:10
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=2 / 9, random_state=42)

    # Torch-Tensoren
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    #Hyperparameter
    learning_rates = [0.1, 0.01, 0.001]
    optimizers = ['Adam', 'SGD', 'RMSprop']
    criterions = {'MSE': nn.MSELoss(), 'MAE': nn.L1Loss()}

    best_val_loss = float('inf')
    best_test_loss = float('inf')
    best_model = None
    best_history = None
    best_config = {}

    for lr in learning_rates:
        for opt_name in optimizers:
            for crit_name, criterion in criterions.items():
                model = DNN(input_size=7, hidden_sizes=[128, 256, 128], output_size=7)

                if opt_name == 'Adam':
                    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
                elif opt_name == 'SGD':
                    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=1e-4)
                elif opt_name == 'RMSprop':
                    optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=1e-4)

                #Train
                history, val_loss = model.train_model(X_train, y_train, X_val, y_val, optimizer, criterion, epochs=100)

                # Test
                test_loss = model.test_model(X_test, y_test, criterions[crit_name])

                if test_loss < best_test_loss:
                    best_val_loss = val_loss
                    best_test_loss = test_loss
                    best_model = model
                    best_history = history
                    best_config = {
                        'lr': lr,
                        'optimizer': opt_name,
                        'loss': crit_name
                    }

    # Save Output
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)

    # Save Model
    torch.save(best_model.state_dict(), os.path.join(output_dir, 'best_model.pt'))

    # Plot loss function
    plt.plot(best_history['train_loss'], label='Train Loss')
    plt.plot(best_history['val_loss'], label='Val Loss')
    plt.title('Loss Curve (Best Model)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_curve.png'))
    plt.close()


    # Save Information
    with open(os.path.join(output_dir, 'info.txt'), 'w') as f:
        f.write("Best Hyperparameters:\n")
        f.write(f"Learning Rate: {best_config['lr']}\n")
        f.write(f"Optimizer: {best_config['optimizer']}\n")
        f.write(f"Loss Function: {best_config['loss']}\n")
        f.write(f"Validation Loss: {best_val_loss:.6f}\n")
        f.write(f"Test Loss: {best_test_loss:.6f}\n")

    # ZIP erstellen
    with zipfile.ZipFile('submission.zip', 'w') as zipf:
        zipf.write(os.path.join(output_dir, 'best_model.pt'), arcname='best_model.pt')
        zipf.write(os.path.join(output_dir, 'loss_curve.png'), arcname='loss_curve.png')
        zipf.write(os.path.join(output_dir, 'info.txt'), arcname='info.txt')

    print("Done! Submission Data saved in folder: submission")