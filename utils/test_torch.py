# %%

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

class LogisticModel(nn.Module):
    def __init__(self):
        super(LogisticModel, self).__init__()

        # Parameters for visual inputs (left and right)
        self.v_L = nn.Parameter(torch.tensor(1.0))   # Coefficient for visual left (visL)
        self.v_R = nn.Parameter(torch.tensor(1.0))   # Coefficient for visual right (visR)
        self.gamma = nn.Parameter(torch.tensor(1.0))  # Power for visual inputs

        # Parameters for auditory inputs (left and right)
        self.a_L = nn.Parameter(torch.tensor(1.0))   # Coefficient for auditory left (audL)
        self.a_R = nn.Parameter(torch.tensor(1.0))   # Coefficient for auditory right (audR)

        # Bias term for S
        self.b = nn.Parameter(torch.tensor(0.0))      # Bias for S

        # Parameters for optogenetic inputs (left and right)
        self.v_L_o = nn.Parameter(torch.tensor(1.0))  # Coefficient for visual opto left (visL_opto)
        self.v_R_o = nn.Parameter(torch.tensor(1.0))  # Coefficient for visual opto right (visR_opto)
        self.gamma_O = nn.Parameter(torch.tensor(1.0)) # Power for visual opto inputs

        self.a_L_o = nn.Parameter(torch.tensor(1.0))  # Coefficient for auditory opto left (audL_opto)
        self.a_R_o = nn.Parameter(torch.tensor(1.0))  # Coefficient for auditory opto right (audR_opto)

        # Bias term for S_O
        self.b_o = nn.Parameter(torch.tensor(0.0))    # Bias for S_O

    def forward(self, X):
        # Extract inputs
        V_L = X[:, 0]  # Assuming visL is the first column
        V_R = X[:, 1]  # Assuming visR is the second column
        A_L = X[:, 2]  # Assuming audL is the third column
        A_R = X[:, 3]  # Assuming audR is the fourth column
        O = X[:, 4]    # Assuming bias_opto is the fifth column

        # Power visual parameters to gamma
        V_L_ = V_L ** self.gamma
        V_R_ = V_R ** self.gamma 

        S = (self.v_L * V_L_ +
             self.v_R * V_R_ +
             self.a_L * A_L +
             self.a_R * A_R + 
             self.b)

        S_opto = (self.v_L_o * V_L_ * O +
                  self.v_R_o * V_R_ * O +
                  self.a_L_o * A_L * O  +
                  self.a_R_o * A_R * O +
                  self.b_o)

        # Calculate logits for probabilities
        logOdds = S + S_opto
        p_R = torch.sigmoid(logOdds)

        return p_R

# Function to train the PyTorch model
def fit_pytorch_model(X_train, y_train, X_test, epochs=1000, lr=0.01, batch_size=100):
    # Convert to Tensor
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
    
    # Create a DataLoader for batching
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize the model, loss function, and optimizer
    model = LogisticModel()
    criterion = nn.BCELoss()  # Binary cross-entropy loss for classification
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    model.train()
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            # Forward pass: compute model predictions
            y_pred = model(inputs).squeeze()  # Ensure the output shape matches labels

            # Compute loss
            loss = criterion(y_pred, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    # Testing phase
    model.eval()  # Evaluation mode disables dropout, etc.
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
        y_pred_test = model(X_test_tensor).numpy()
    
    # Return the trained model and test predictions
    return model, y_pred_test   

