import requests
import zipfile
import io
import os
import json
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.optimizers import Adam

# Configuration
REPO_URL = "https://github.com/fchollet/ARC-AGI/archive/refs/heads/master.zip"
DATA_PATH_IN_ZIP = "ARC-AGI-master/data/"
EXTRACT_TO = "arc_data_temp"  # Local path for extraction

# Create extraction directory if it doesn't exist
os.makedirs(EXTRACT_TO, exist_ok=True)

# Download repository
print("Downloading repository...")
response = requests.get(REPO_URL)
response.raise_for_status()

# Extract only the data folder
print("Extracting files...")
with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
    # Get all members in the data directory
    data_members = [m for m in zip_ref.namelist()
                    if m.startswith(DATA_PATH_IN_ZIP) and not m.endswith('/')]
    zip_ref.extractall(EXTRACT_TO, members=data_members)

# Local path adjustment
data_path = os.path.join(EXTRACT_TO, DATA_PATH_IN_ZIP)

# Verify extraction
print("Extracted files:")
for root, dirs, files in os.walk(data_path):
    for name in files:
        print(os.path.join(root, name))

# Find subfolders
subfolders = [f for f in os.listdir(data_path)
              if os.path.isdir(os.path.join(data_path, f))]

if len(subfolders) != 2:
    raise ValueError(f"Expected 2 subfolders, found {len(subfolders)}")

# Create variables for file paths
folder1, folder2 = subfolders
folder1_path = os.path.join(data_path, folder1)
folder2_path = os.path.join(data_path, folder2)

# Get all files in each folder
folder1_files = [os.path.join(folder1_path, f) for f in os.listdir(folder1_path)]
folder2_files = [os.path.join(folder2_path, f) for f in os.listdir(folder2_path)]

# Helper function to load data from JSON files
def load_data(file_paths):
    data = []
    for file_path in file_paths:
        with open(file_path, 'r') as file:
            data.append(json.load(file))
    return data

# Helper function to find the maximum shape
def find_max_shape(data):
    max_rows, max_cols = 0, 0
    for item in data:
        for example in item['train'] + item['test']:
            rows, cols = len(example['input']), len(example['input'][0])
            max_rows = max(max_rows, rows)
            max_cols = max(max_cols, cols)
    return max_rows, max_cols

# Helper function to pad sequences
def pad_sequence(sequence, max_shape, pad_value=0):
    if not sequence:  # Check if the sequence is empty
        return np.full(max_shape, pad_value)
    padded_sequence = np.full(max_shape, pad_value)
    rows, cols = len(sequence), len(sequence[0])
    padded_sequence[:rows, :cols] = sequence
    return padded_sequence

# Helper function to preprocess the data
def preprocess_data(data):
    X_train, y_train, X_test, y_test = [], [], [], []
    max_shape = find_max_shape(data)
    for item in data:
        for train_example in item['train']:
            X_train.append(pad_sequence(train_example['input'], max_shape))
            y_train.append(pad_sequence(train_example['output'], max_shape))
        for test_example in item['test']:
            X_test.append(pad_sequence(test_example['input'], max_shape))
            y_test.append(pad_sequence(test_example['output'], max_shape))
    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

# Load and preprocess data from folder2_files
folder2_data = load_data(folder2_files)
X_train, y_train, X_test, y_test = preprocess_data(folder2_data)

# Multi-Label Classification model
mlb = MultiLabelBinarizer()
y_train_mlb = mlb.fit_transform([tuple(map(tuple, y)) for y in y_train])
y_test_mlb = mlb.transform([tuple(map(tuple, y)) for y in y_test])

def create_multi_label_model(input_shape, num_classes):
    model = Sequential([
        Dense(128, activation='relu', input_shape=input_shape),
        Dense(num_classes, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

multi_label_model = create_multi_label_model((X_train.shape[1] * X_train.shape[2],), y_train_mlb.shape[1])

# Neural Network (NN) model
def create_nn_model(input_shape, num_classes):
    model = Sequential([
        Dense(128, activation='relu', input_shape=input_shape),
        Dense(num_classes, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

nn_model = create_nn_model((X_train.shape[1] * X_train.shape[2],), y_train_mlb.shape[1])

# Convolutional Neural Network (CNN) model
def create_cnn_model(input_shape):
    model = Sequential([
        Conv2D(32, kernel_size=3, activation='relu', input_shape=input_shape),
        Flatten(),
        Dense(y_train_mlb.shape[1], activation='sigmoid')
    ])
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

cnn_model = create_cnn_model((X_train.shape[1], X_train.shape[2], 1))

# Flatten the input data for the NN and Multi-Label models
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# Create folder1_files2 as a temporary variable by copying folder1_files and removing test content
folder1_files2_temp = []
for file_path in folder1_files:
    with open(file_path, 'r') as file:
        data = json.load(file)
    # Remove content of the test section but keep the test label
    for test_example in data['test']:
        test_example['output'] = []  # Empty the output section
    folder1_files2_temp.append(data)

# Load and preprocess data from folder1_files2_temp
X_train2, y_train2, X_test2, y_test2 = preprocess_data(folder1_files2_temp)

# Flatten the input data for the NN model
X_train2_flat = X_train2.reshape(X_train2.shape[0], -1)
X_test2_flat = X_test2.reshape(X_test2.shape[0], -1)

# Multi-Label Transformation for folder1_files2_temp
y_train2_mlb = mlb.transform([tuple(map(tuple, y)) for y in y_train2])
y_test2_mlb = mlb.transform([tuple(map(tuple, y)) for y in y_test2])

# Training loop (30 iterations)
for iteration in range(30):
    # Train Multi-Label Classification model
    multi_label_model.fit(X_train_flat, y_train_mlb, epochs=5, verbose=1)

    # Get predictions from Multi-Label Classification model
    y_train_preds = multi_label_model.predict(X_train_flat)

    # Train NN model
    nn_model.fit(X_train_flat, y_train_preds, epochs=5, verbose=1)

    # Train CNN model
    cnn_model.fit(X_train[..., np.newaxis], y_train_preds, epochs=5, verbose=1)

    # Adjust Multi-Label Classification based on CNN model
    y_train_preds_cnn = cnn_model.predict(X_train[..., np.newaxis])
    multi_label_model.fit(X_train_flat, y_train_preds_cnn, epochs=5, verbose=1)

    # Adjust NN model weights based on new categories
    y_train_preds_new = multi_label_model.predict(X_train_flat)
    nn_model.fit(X_train_flat, y_train_preds_new, epochs=5, verbose=1)

# Train the final NN model on folder1_files2_temp
nn_model.fit(X_train2_flat, y_train2_mlb, epochs=5, verbose=1)

# Calculate accuracy of final NN model on folder1_files2_temp
y_test2_preds = nn_model.predict(X_test2_flat)
accuracy2 = accuracy_score(y_test2_mlb, y_test2_preds.round())
print(f'Accuracy on folder1_files2: {accuracy2*100:.2f}%')

# Calculate exact match ratio between folder1_files2_temp and folder1_files
exact_matches = sum(np.array_equal(y1, y2) for y1, y2 in zip(y_test2_preds.round(), y_test_mlb))
exact_match_ratio = exact_matches / len(y_test_mlb)
print(f'Exact match ratio between folder1_files2 and folder1_files: {exact_match_ratio*100:.2f}%')

# Cross-validation between folder1_files2_temp and folder2_files
# Load and preprocess data from folder1_files2_temp (already done)
# Load and preprocess data from folder2_files (already done)

# Flatten the input data for the cross-validation
X_train2_flat = X_train2.reshape(X_train2.shape[0], -1)
X_test2_flat = X_test2.reshape(X_test2.shape[0], -1)

# Train the final NN model on folder1_files2_temp
nn_model.fit(X_train2_flat, y_train2_mlb, epochs=5, verbose=1)

# Calculate accuracy of final NN model on folder2_files
y_test_preds = nn_model.predict(X_test_flat)
accuracy = accuracy_score(y_test_mlb, y_test_preds.round())
print(f'Accuracy on folder2_files: {accuracy*100:.2f}%')

# Calculate accuracy of final NN model on folder1_files2_temp
y_test2_preds = nn_model.predict(X_test2_flat)
accuracy2 = accuracy_score(y_test2_mlb, y_test2_preds.round())
print(f'Accuracy on folder1_files2: {accuracy2*100:.2f}%')
