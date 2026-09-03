Credit Card Fraud Detection Using Neural Networks

Project overview
This project uses deep learning to identify fraudulent credit-card transactions. It compares two PyTorch neural-network architectures trained on the original imbalanced dataset without using SMOTE or any other oversampling method.
The class imbalance is handled using a weighted BCEWithLogitsLoss. This gives the rare fraudulent transactions more importance during training while leaving the original dataset unchanged.

Dataset
The dataset contains credit-card transactions made by European cardholders during two days in September 2013.
Total transactions: 284,807
Legitimate transactions: 284,315
Fraudulent transactions: 492
Fraud rate: approximately 0.172%

The dataset is highly imbalanced because fraudulent transactions represent only a very small percentage of all transactions.
Dataset columns
Columns
Description
Time
Seconds between each transaction and the first transaction
V1–V28
Anonymized numerical features produced using PCA
Amount
Transaction amount
Class
Target: 0 means legitimate and 1 means fraud

The dataset file is not included in the repository because of its size. Download creditcard.csv separately and place it beside the Python script.

Project objective

The objective is to train and compare two neural networks and determine which one identifies fraudulent transactions more effectively.

The models are evaluated using:

Precision

Recall

F1-score

Area Under the Precision–Recall Curve (AUPRC)

AUPRC is the main comparison metric because accuracy can be misleading for a dataset in which more than 99% of transactions are legitimate.

Neural-network models

SmallMLP

30 inputs → 64 neurons → 32 neurons → 1 output

The smaller network uses ReLU activation and dropout to reduce overfitting.

DeepMLP

30 inputs → 128 neurons → 64 neurons → 32 neurons → 1 output

The deeper network uses ReLU activation, batch normalization and dropout.

The final layer of each network returns one raw value called a logit. A sigmoid function is applied later when fraud probabilities are needed.

Handling class imbalance

The project does not use SMOTE. Instead, it calculates a positive-class weight from the training data:

pos_weight_value = number_of_legitimate_transactions / number_of_fraud_transactions

This weight is passed to:

nn.BCEWithLogitsLoss(pos_weight=pos_weight)

As a result, an error involving a fraudulent transaction has a larger effect on the loss than an error involving a legitimate transaction.

BCEWithLogitsLoss combines the sigmoid operation and binary cross-entropy calculation in a stable way. The model therefore returns raw logits and does not need a sigmoid layer inside the architecture.

Data preparation

The dataset is divided into:

70% training data

15% validation data

15% test data

Stratified splitting is used to preserve approximately the same fraud proportion in all three sets.

StandardScaler is fitted only on the training data. The fitted scaler is then used to transform the validation and test data. This prevents information from the validation or test set from leaking into training.

Training process

Each model is trained using:

Adam optimizer

Learning rate of 0.001

Batch size of 1024

Maximum of 25 epochs

Early stopping patience of 5 epochs

Validation AUPRC is checked after every epoch. The weights that achieve the highest validation AUPRC are kept. Training stops early if validation AUPRC does not improve for five consecutive epochs.

The validation set is also used to select the probability threshold that produces the best F1-score. That threshold is then applied once to the untouched test set.

Evaluation metrics

Precision

Precision measures how many transactions predicted as fraud were actually fraudulent:

Precision = True Positives / (True Positives + False Positives)

High precision means the system produces fewer false fraud alerts.

Recall

Recall measures how many actual fraudulent transactions were detected:

Recall = True Positives / (True Positives + False Negatives)

High recall means fewer fraudulent transactions are missed.

F1-score

F1-score combines precision and recall into one value:

F1 = 2 × (Precision × Recall) / (Precision + Recall)

AUPRC

AUPRC summarizes the precision–recall relationship across different probability thresholds. It is particularly suitable for fraud detection because the fraud class is extremely rare.

Project structure

credit-card-fraud/
├── credit_card_fraud.py
├── creditcard.csv
└── README.md

Files produced after running the script include:

SmallMLP_training_history.csv
DeepMLP_training_history.csv
SmallMLP_weights.pt
DeepMLP_weights.pt
neural_network_comparison.csv
neural_network_precision_recall_curves.png

Installation

Create and activate a virtual environment if desired, then install the packages:

python -m pip install torch pandas numpy scikit-learn matplotlib

Confirm that PyTorch is installed:

python -c "import torch; print(torch.__version__)"

Running the project

Place creditcard.csv in the same folder as credit_card_fraud.py, then run:

python credit_card_fraud.py

If the dataset is stored somewhere else, change the path inside the script:

data_path = Path(r"D:\credit card fraud\data\creditcard.csv")

Results

After training, add the values from neural_network_comparison.csv to this table:

Model

Threshold

Precision

Recall

F1

AUPRC

SmallMLP

—

—

—

—

—

DeepMLP

—

—

—

—

—

The model with the highest test AUPRC provides the stronger overall ranking of fraudulent transactions. Precision, recall and F1 should also be considered because they describe performance at the selected classification threshold.

Limitations

The dataset covers only two days of transactions from 2013.

The anonymized PCA features are difficult to interpret directly.

Fraud patterns can change over time.

Weighted loss can improve attention to fraud, but it does not guarantee that every fraud will be detected.

Results from this educational experiment should not be treated as a production fraud-detection system.

Future improvements

Compare different decision thresholds.

Add model explainability methods.

Test the models on newer transaction data.

Add probability calibration.

Monitor changes in fraud patterns over time.

Disclaimer

This project is intended for learning and research. A real financial fraud-detection system would require secure data handling, model monitoring, human review and regular validation.
