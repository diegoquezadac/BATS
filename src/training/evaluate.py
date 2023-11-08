import torch

def evaluate_model(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, device: torch.device, metrics: dict):
    """
    Evaluates the performance of a PyTorch model on a given dataset using specified metrics.

    Args:
        model: The PyTorch model to be evaluated.
        data_loader: DataLoader for the test or validation dataset.
        device: The device (CPU or GPU) to perform the evaluation on.
        metrics: A dictionary of metrics to be calculated. Each entry should be a string
                 representing the metric name and the corresponding function that takes
                 the true and predicted values as arguments.

    Returns:
        A dictionary with the calculated metric values.
    """
    model.to(device)
    model.eval()  # Set the model to evaluation mode

    metric_values = {metric_name: 0 for metric_name in metrics.keys()}
    total_samples = 0

    with torch.no_grad():  # Disable gradient calculation for efficiency
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            #predictions = outputs.argmax(dim=1) if not isinstance(outputs, tuple) else outputs[0].argmax(dim=1)
            
            # Update the total number of samples
            total_samples += labels.size(0)

            # Calculate each metric and accumulate the values
            for metric_name, metric_fn in metrics.items():
                metric_values[metric_name] += metric_fn(outputs, labels).item()

    # Average the metric values over all samples
    for metric_name in metrics.keys():
        metric_values[metric_name] /= total_samples

    return metric_values