import torch


def calculate_correct_num(y_true, y_pred):
    """Calculate accuracy given the true labels and predicted labels."""
    prediction = y_pred.argmax(dim=1)
    truth = y_true.argmax(dim=1)
    num_correct = torch.eq(prediction, truth).sum().float().item()

    return num_correct
