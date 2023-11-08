import torch
from src.training.early_stopping import CustomEarlyStopping

def train_model(model: torch.nn.Module, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader, epochs: int, criterion: torch.nn, optimizer: torch.optim, early_stopping: CustomEarlyStopping = None):
    """
    Trains a PyTorch model with the given data loaders, criterion, and optimizer.

    Args:
        model: The PyTorch model to be trained.
        train_loader: DataLoader for the training data.
        val_loader: DataLoader for the validation data.
        epochs: The number of epochs to train the model.
        criterion: The loss function used to evaluate the model's performance.
        optimizer: The optimization algorithm used to update the model's weights.
        device: The device (CPU or GPU) to train the model on.
        early_stopping: An optional instance of CustomEarlyStopping to stop training early if validation loss does not improve.

    Returns:
        The trained model with the best validation loss if early stopping is used, otherwise the model after the last training epoch.
    """
    # Iterate over the entire dataset for a specified number of epochs
    for epoch in range(epochs):
        # Set the model to training mode
        model.train()
        running_loss = 0.0

        # Iterate over the training data
        for inputs, labels in train_loader:

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass: compute the model output
            outputs = model(inputs)
            # Compute the loss
            loss = criterion(outputs, labels)

            # Backward pass: compute the gradient of the loss with respect to model parameters
            loss.backward()
            # Perform a single optimization step to update the model parameters
            optimizer.step()

            # Accumulate the training loss
            running_loss += loss.item()

        # Set the model to evaluation mode
        model.eval()
        val_loss = 0.0
        # Disable gradient calculation for validation to save memory and computations
        with torch.no_grad():
            # Iterate over the validation data
            for inputs, labels in val_loader:
                # Compute the model output
                outputs = model(inputs)
                # Compute the loss
                loss = criterion(outputs, labels)
                # Accumulate the validation loss
                val_loss += loss.item()

        # Print the training and validation loss statistics
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}')

        # Early Stopping check
        if early_stopping is not None:
            # Call the early stopping logic
            early_stopping(val_loss/len(val_loader), model)
            if early_stopping.early_stop:
                # If early stop condition was met, print a message and break the training loop
                print("Early stopping")
                # Load the best model state (with the lowest validation loss)
                model.load_state_dict(early_stopping.best_state)
                break

    # Return the trained model
    return model
