from typing import Literal
import torch

class CustomEarlyStopping:
    """
    Implements early stopping to stop the training when the validation loss does not improve.

    Attributes:
        patience (int): Number of epochs to wait after min has been hit. After this
                        number of epochs, training will stop.
        delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        verbose (bool): If True, prints a message for each validation loss improvement.
        mode (Literal['min', 'max']): If 'min', training will stop when the quantity
                                      monitored has stopped decreasing; if 'max', it will
                                      stop when the quantity monitored has stopped increasing.
        counter (int): Counter that keeps track of how many epochs have passed without
                       improvement.
        best_score (Optional[float]): The best score reached by the model, used to determine
                                      when to save the model state.
        early_stop (bool): Flag that indicates if the early stopping condition was met.
        best_state (Optional[dict]): The best model state, this gets loaded back into the model
                                     when training is finished.
    """
    
    def __init__(self, patience: int = 10, delta: float = 0.0, verbose: bool = False, mode: Literal['min', 'max'] = 'min'):
        """
        Initializes the CustomEarlyStopping instance.

        Args:
            patience (int): Number of epochs with no improvement after which training will be stopped.
            delta (float): The minimum change to qualify as an improvement.
            verbose (bool): If True, prints a message for each improvement.
            mode (Literal['min', 'max']): Decides whether a lower or higher score is better.
        """
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_state = None
        self.mode = mode
    
    def __call__(self, val_loss: float, model: torch.nn.Module):
        """
        Call method that compares the current validation loss with the best one and updates the best score.

        Args:
            val_loss (float): The current validation loss.
            model (torch.nn.Module): The model being trained.
        """
        # Adjust the sign of the score based on the optimization mode
        score = -val_loss if self.mode == 'min' else val_loss

        # Check if this is the first iteration (best_score is still None)
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        # Check if the current score is worse (plus the delta) than the best score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # If current score is better, save the model checkpoint
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss: float, model: torch.nn.Module):
        """
        Saves the current model as a checkpoint if the current validation loss is lower than the best one.

        Args:
            val_loss (float): The current validation loss.
            model (torch.nn.Module): The model being trained.
        """
        if self.verbose:
            print(f'Validation loss decreased ({self.best_score:.6f} --> {val_loss:.6f}). Saving model ...')
        # Save the model state
        self.best_state = model.state_dict()

    def load_best_state(self, model):
        """
        Loads the best model state into the given model.

        Args:
            model (torch.nn.Module): The model to load the state into.

        Returns:
            model (torch.nn.Module): The model with the best state loaded.
        """
        # Ensure there is a best state to load from
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
        return model
