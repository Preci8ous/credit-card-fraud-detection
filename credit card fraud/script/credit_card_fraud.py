# Used to save a copy of the best model during training
from copy import deepcopy
from pathlib import Path

import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
BATCH_SIZE = 1024
MAX_EPOCHS = 25
PATIENCE = 5
LEARNING_RATE = 0.001
TARGET = "Class"


def set_seed(seed=SEED):
    """Set the random seed so that results are reproducible."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_and_split_data(file_path):
    """Load, check, split and scale the dataset."""

    # Read the credit-card CSV file
    data = pd.read_csv(file_path)

    features = (
        ["Time"]
        + [f"V{i}" for i in range(1, 29)]
        + ["Amount"]
    )

    expected = features + [TARGET]
    missing = set(expected) - set(data.columns)
    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    # Keep only the columns needed by the project
    data = data[expected].copy()

    # Make sure every column contains numerical values
    for column in expected:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    # Invalid values become missing, so check for them
    if data.isna().any().any():
        raise ValueError(
            "The dataset contains missing or non-numeric values."
        )

    # Class must contain only 0 and 1
    if not set(data[TARGET].unique()).issubset({0, 1}):
        raise ValueError(
            "Class must contain only 0 and 1."
        )

    # X contains the 30 model inputs
    X = data[features].to_numpy(
        dtype=np.float32
    )

    # y contains the target:
    # 0 means legitimate and 1 means fraud
    y = data[TARGET].to_numpy(
        dtype=np.float32
    )

    # First, separate 15% of the data for final testing
    X_train_val, X_test, y_train_val, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.15,
            stratify=y,
            random_state=SEED,
        )
    )

    # Divide the remaining data into training and validation sets
    X_train, X_val, y_train, y_val = (
        train_test_split(
            X_train_val,
            y_train_val,
            test_size=0.17647,
            stratify=y_train_val,
            random_state=SEED,
        )
    )
    # Create the feature scaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(
        X_train
    ).astype(np.float32)

    X_val = scaler.transform(
        X_val
    ).astype(np.float32)

    X_test = scaler.transform(
        X_test
    ).astype(np.float32)
    # Return all three parts of the dataset
    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    )

def make_loader(X, y, shuffle=False):
    """Convert the data into PyTorch batches."""
    dataset = TensorDataset(
        torch.tensor(
            X,
            dtype=torch.float32,
        ),
        torch.tensor(
            y,
            dtype=torch.float32,
        ),
    )

    # Divide the dataset into batches
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
    )

class SmallMLP(nn.Module):
    """The first and smaller neural network."""
    def __init__(self, input_size=30):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        """Pass input data through the network."""
        # The network produces raw logits
        output = self.network(x)
        # Remove the unnecessary second dimension
        return output.squeeze(1)

class DeepMLP(nn.Module):
    """The second and deeper neural network."""
    def __init__(self, input_size=30):
        super().__init__()
        # Define the deeper network
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        """Pass input data through the deeper network."""
        output = self.network(x)
        return output.squeeze(1)
    
# Do not calculate gradients while making predictions
@torch.no_grad()
def predict_probabilities(model, loader, device):
    """Use a trained model to calculate fraud probabilities."""
    # Put the model into evaluation mode
    model.eval()
    # These lists will store the true labels and predictions
    labels, probabilities = [], []
    # Process one batch at a time
    for features, targets in loader:
        # Move the features to the CPU or GPU
        features = features.to(device)
        # Get the raw model outputs
        logits = model(features)

        # Convert the logits into values between 0 and 1
        fraud_probabilities = torch.sigmoid(logits)

        # Move probabilities back to the CPU and save them
        probabilities.extend(
            fraud_probabilities.cpu().numpy()
        )

        # Save the true target values
        labels.extend(targets.numpy())

    # Convert the two lists into NumPy arrays
    return (
        np.asarray(labels, dtype=int),
        np.asarray(probabilities),
    )


def best_f1_threshold(labels, probabilities):
    """Find the validation threshold that gives the best F1-score."""

    # Calculate precision and recall at different thresholds
    precision, recall, thresholds = (
        precision_recall_curve(
            labels,
            probabilities,
        )
    )

    # Use 0.5 if no thresholds can be calculated
    if len(thresholds) == 0:
        return 0.5

    # Calculate the F1-score at every available threshold
    f1_values = (
        2
        * precision[:-1]
        * recall[:-1]
        / (
            precision[:-1]
            + recall[:-1]
            + 1e-12
        )
    )

    # Find the position of the highest F1-score
    best_position = np.argmax(f1_values)

    # Return the corresponding threshold
    return float(thresholds[best_position])


def calculate_metrics(
    labels,
    probabilities,
    threshold,
):
    """Calculate the four project evaluation metrics."""

    # Convert probabilities into class predictions
    predictions = (
        probabilities >= threshold
    ).astype(int)

    # Calculate and return the requested metrics
    return {
        "Precision": precision_score(
            labels,
            predictions,
            zero_division=0,
        ),
        "Recall": recall_score(
            labels,
            predictions,
            zero_division=0,
        ),
        "F1": f1_score(
            labels,
            predictions,
            zero_division=0,
        ),
        "AUPRC": average_precision_score(
            labels,
            probabilities,
        ),
    }


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    device,
):
    """Train one model and use early stopping."""

    # Adam updates the model weights during training
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # Save a copy of the model's starting weights
    best_state = deepcopy(
        model.state_dict()
    )

    # Keep track of the best validation result
    best_auprc = -np.inf

    # Counts epochs that do not improve the model
    waiting = 0

    # Stores the training results from each epoch
    history = []

    # Start the training epochs
    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):
        # Put the model into training mode
        model.train()

        # Used to calculate average training loss
        loss_sum = 0.0
        examples = 0

        # Process every training batch
        for features, targets in train_loader:
            # Move the batch to the CPU or GPU
            features = features.to(device)
            targets = targets.to(device)

            # Remove gradients left from the previous batch
            optimizer.zero_grad()

            # Pass the features through the model
            logits = model(features)

            # Compare predictions with the true targets
            loss = criterion(
                logits,
                targets,
            )

            # Calculate the gradients
            loss.backward()

            # Update the model weights
            optimizer.step()

            # Add this batch's loss to the total
            loss_sum += (
                loss.item() * len(targets)
            )

            # Count how many examples were processed
            examples += len(targets)

        # Get predictions for the validation set
        labels, probabilities = (
            predict_probabilities(
                model,
                val_loader,
                device,
            )
        )

        # Calculate validation AUPRC
        validation_auprc = (
            average_precision_score(
                labels,
                probabilities,
            )
        )

        # Calculate the average loss for this epoch
        mean_loss = loss_sum / examples

        # Save this epoch's results
        history.append(
            {
                "Epoch": epoch,
                "Train Loss": mean_loss,
                "Validation AUPRC": validation_auprc,
            }
        )

        # Display the current training progress
        print(
            f"Epoch {epoch:02d} "
            f"| loss={mean_loss:.5f} "
            f"| val AUPRC={validation_auprc:.5f}"
        )

        # Check whether the model has improved
        if validation_auprc > best_auprc + 1e-5:
            # Save the improved validation score
            best_auprc = validation_auprc

            # Save a copy of the improved model
            best_state = deepcopy(
                model.state_dict()
            )

            # Reset the early-stopping counter
            waiting = 0

        else:
            # The model did not improve
            waiting += 1

            # Stop after several epochs without improvement
            if waiting >= PATIENCE:
                print("Early stopping")
                break

    # Restore the model that achieved the best validation AUPRC
    model.load_state_dict(best_state)

    # Return the best model and its training history
    return model, pd.DataFrame(history)


def main():
    """Run the complete fraud-detection experiment."""

    # Set the random seed
    set_seed()

    # Use the GPU when available, otherwise use the CPU
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    # Location of the credit-card dataset
    data_path = Path(r"D:\credit card fraud\data\creditcard.csv" )

    # Load and prepare the dataset
    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = load_and_split_data(data_path)
    # Display the number of records in each part
    print(
        f"Train: {len(y_train):,} "
        f"| Validation: {len(y_val):,} "
        f"| Test: {len(y_test):,}"
    )
    # Display the number of fraud examples in training
    print(
        f"Training frauds: "
        f"{int(y_train.sum()):,} "
        f"of {len(y_train):,}"
    )
    # Create loaders for the three datasets
    train_loader = make_loader(
        X_train,
        y_train,
        shuffle=True,
    )
    val_loader = make_loader(
        X_val,
        y_val,
    )
    test_loader = make_loader(
        X_test,
        y_test,
    )
    # Count legitimate transactions in training
    negatives = int(
        (y_train == 0).sum()
    )
    # Count fraudulent transactions in training
    positives = int(
        (y_train == 1).sum()
    )
    # Calculate how much more common legitimate transactions are
    pos_weight_value = negatives / positives

    # Convert the weight into a PyTorch tensor
    pos_weight = torch.tensor(
        pos_weight_value,
        dtype=torch.float32,
        device=device,
    )
    # Create the weighted binary classification loss
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    print(
        "BCEWithLogitsLoss pos_weight:",
        f"{pos_weight_value:.2f}",
    )

    # Store the two models that will be compared
    model_classes = {
        "SmallMLP": SmallMLP,
        "DeepMLP": DeepMLP,
    }

    # Store final metrics and graph values
    results = []
    curves = {}

    # Train and evaluate each model
    for name, model_class in model_classes.items():
        print(f"\nTraining {name}")

        # Create the current model
        model = model_class(
            X_train.shape[1]
        ).to(device)

        # Train the model
        model, history = train_model(
            model,
            train_loader,
            val_loader,
            criterion,
            device,
        )

        # Save its training history
        history.to_csv(
            f"{name}_training_history.csv",
            index=False,
        )

        # Get validation probabilities
        val_labels, val_probabilities = (
            predict_probabilities(
                model,
                val_loader,
                device,
            )
        )

        # Choose the threshold using validation data
        threshold = best_f1_threshold(
            val_labels,
            val_probabilities,
        )

        # Get final predictions for the untouched test data
        test_labels, test_probabilities = (
            predict_probabilities(
                model,
                test_loader,
                device,
            )
        )

        # Calculate the final test metrics
        metrics = calculate_metrics(
            test_labels,
            test_probabilities,
            threshold,
        )

        # Add this model's results to the list
        results.append(
            {
                "Model": name,
                "Threshold": threshold,
                **metrics,
            }
        )

        # Calculate values needed for the PR curve
        precision, recall, _ = (
            precision_recall_curve(
                test_labels,
                test_probabilities,
            )
        )

        # Save the curve values
        curves[name] = (
            precision,
            recall,
            metrics["AUPRC"],
        )

        # Save the trained model weights
        torch.save(
            model.state_dict(),
            f"{name}_weights.pt",
        )

    # Convert the model results into a table
    results_df = pd.DataFrame(results)

    # Put the model with the highest AUPRC first
    results_df = results_df.sort_values(
        "AUPRC",
        ascending=False,
    )

    # Save the comparison table
    results_df.to_csv(
        "neural_network_comparison.csv",
        index=False,
    )

    # Display the results
    print("\nFinal test-set results:")

    print(
        results_df
        .round(5)
        .to_string(index=False)
    )

    # Create the precision-recall graph
    plt.figure(figsize=(8, 6))

    # Add a curve for each neural network
    for name, (
        precision,
        recall,
        auprc,
    ) in curves.items():
        plt.plot(
            recall,
            precision,
            label=(
                f"{name} "
                f"(AUPRC={auprc:.3f})"
            ),
        )

    # Add graph labels and title
    plt.xlabel("Recall")
    plt.ylabel("Precision")

    plt.title(
        "Credit-card fraud precision-recall curves"
    )

    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    # Save the graph
    plt.savefig(
        "neural_network_precision_recall_curves.png",
        dpi=180,
    )

    # Close the graph
    plt.close()

    print(
        "\nSaved results, curves, "
        "histories and model weights"
    )


# Run main() only when this script is executed directly
if __name__ == "__main__":
    main()