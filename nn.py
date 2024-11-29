import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from candy import Candy
from analysis import *
from torch.utils.data import random_split
import matplotlib.pyplot as plt

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Example unique IDs and types
unique_ids = [1, 2, 3, 4, 5, 6, 7]
unique_types = ["normal", "sachet", "raye_hor", "raye_ver", "disco"]

# Mapping for one-hot encoding
id_to_index = {id_: idx for idx, id_ in enumerate(unique_ids)}
type_to_index = {type_: idx for idx, type_ in enumerate(unique_types)}

# One-hot encode a single Candy object
def one_hot_encode_candy(candy):
    id_one_hot = np.zeros(len(unique_ids))
    type_one_hot = np.zeros(len(unique_types))
    id_one_hot[id_to_index[candy.id]] = 1
    type_one_hot[type_to_index.get(candy.type, 0)] = 1  # Default to first index if type missing
    return np.concatenate([id_one_hot, type_one_hot])

# Convert a 7x7 grid of Candy objects to a numerical 3D array
def preprocess_board(board):
    grid_size = (7, 7)
    encoded_board = np.zeros((*grid_size, len(unique_ids) + len(unique_types)))
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            encoded_board[i, j, :] = one_hot_encode_candy(board[i][j])
    return encoded_board


index_to_id = {idx: id_ for id_, idx in id_to_index.items()}
index_to_type = {idx: type_ for type_, idx in type_to_index.items()}

# Decode a single one-hot encoded candy
def decode_candy(encoded_candy):
    id_one_hot = encoded_candy[:len(unique_ids)]
    type_one_hot = encoded_candy[len(unique_ids):]
    
    candy_id = index_to_id[np.argmax(id_one_hot)]
    candy_type = index_to_type[np.argmax(type_one_hot)]
    
    return Candy(candy_id, candy_type)

# Convert a numerical 3D array back to a 7x7 grid of Candy objects
def decode_board(encoded_board):
    grid_size = (7, 7)
    decoded_board = np.empty(grid_size, dtype=object)
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            decoded_board[i, j] = decode_candy(encoded_board[i, j, :])
    return decoded_board



class CandyDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.min_score = dataframe['score'].min()
        self.max_score = dataframe['score'].max()

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        board = self.dataframe.iloc[idx, 0]  # Board state
        score = self.dataframe.iloc[idx, 1]  # Score
        #normalized_score = (score - self.min_score) / (self.max_score - self.min_score)  # Min-Max normalization
        processed_board = preprocess_board(board)
        return torch.tensor(processed_board, dtype=torch.float32), torch.tensor(score, dtype=torch.float32)

class CandyCNN(nn.Module):
    def __init__(self):
        super(CandyCNN, self).__init__()
        in_channels = len(unique_ids) + len(unique_types)  # Total channels from encoding
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=4, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=2, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Dropout layers after the convolutional layers
        self.dropout_conv = nn.Dropout2d(p=0.15)  # Dropout on convolutional feature maps
        
        # Calculate the size after convolutions and pooling
        self._calculate_fc_input_size()

        # Fully connected layers with dropout
        self.fc1 = nn.Linear(self.fc_input_size, 256)  # Adjusted input size
        self.dropout_fc = nn.Dropout(p=0.2)  # Dropout after the first fully connected layer
        self.fc2 = nn.Linear(256, 64)  # Single output for regression
        self.fc3 = nn.Linear(64, 1)  # Single output for regression
        

    def _calculate_fc_input_size(self):
        # Dummy input to calculate output size after convolutions and pooling
        dummy_input = torch.zeros(1, len(unique_ids) + len(unique_types), 7, 7)
        x = self.pool(torch.relu(self.conv1(dummy_input)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        self.fc_input_size = np.prod(x.size()[1:])  # Flattened size after pooling

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.dropout_conv(x)  # Apply dropout to convolutional feature maps
        x = x.reshape(x.size(0), -1) # Flatten for FC layer
        x = torch.relu(self.fc1(x))
        x = self.dropout_fc(x)  # Apply dropout to the first fully connected layer
        x = self.fc2(x)
        x = self.fc3(x)
        return x
    
    
# Modified training function to track training and validation loss
def train_model(model, train_dataloader, val_dataloader, criterion, optimizer, epochs=10):
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_train_loss = 0.0
        for inputs, targets in train_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs.permute(0, 3, 1, 2))  # Adjust dimensions to (N, C, H, W)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()
        
        avg_train_loss = running_train_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()  # Set the model to evaluation mode
        running_val_loss = 0.0
        with torch.no_grad():  # Disable gradient calculation during validation
            for inputs, targets in val_dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs.permute(0, 3, 1, 2))  # Adjust dimensions to (N, C, H, W)
                loss = criterion(outputs.squeeze(), targets)
                running_val_loss += loss.item()

        avg_val_loss = running_val_loss / len(val_dataloader)
        val_losses.append(avg_val_loss)

        # Print the losses
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
    
    # Plot training and validation losses
    plt.plot(range(epochs), train_losses, label='Training Loss')
    plt.plot(range(epochs), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss over Epochs')
    plt.show()



# Function to predict the score for a given board
def predict(board, model):
    model.eval()
    with torch.no_grad():
        processed_board = preprocess_board(board)
        input_tensor = torch.tensor(processed_board, dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2).to(device)  # Shape: (1, C, H, W)
        prediction = model(input_tensor)
        return prediction.item()


def save_model(model, path):
    """
    Save the PyTorch model to the specified path.
    
    Args:
    - model (torch.nn.Module): The model to save.
    - path (str): The path where the model will be saved.
    """
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(path, device):
    """
    Load the PyTorch model from the specified path.
    
    Args:
    - path (str): The path from where the model will be loaded.
    - device (torch.device): The device to load the model onto.
    
    Returns:
    - model (torch.nn.Module): The model with loaded state dict.
    """
    model = CandyCNN()
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    print(f"Model loaded from {path}")
    return model