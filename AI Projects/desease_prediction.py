import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib
import os

# Step 2: Load and Explore the Dataset
def load_and_explore_data(filepath):
    """Load the dataset and provide basic insights."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"The file {filepath} does not exist.")
    
    patient_data = pd.read_csv(filepath)
    print("Dataset loaded successfully!")
    
    print("\nFirst few rows of the dataset:")
    print(patient_data.head())
    
    print("\nMissing values in the dataset:")
    print(patient_data.isnull().sum())
    
    print("\nDataset statistics:")
    print(patient_data.describe())
    
    return patient_data

# Step 3: Preprocess the Data
def preprocess_data(patient_data):
    """Preprocess the data by splitting features and target, and scaling features."""
    features = patient_data.drop("Outcome", axis=1)
    target = patient_data["Outcome"]
    
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print("\nData preprocessing completed!")
    return X_train, X_test, y_train, y_test, scaler

# Step 4: Train and Save a Random Forest Model
def train_random_forest(X_train, y_train):
    """Train a Random Forest model and save it to disk."""
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Save the model
    joblib.dump(rf_model, "random_forest_model.pkl")
    print("\nRandom Forest model trained and saved successfully!")
    return rf_model

# Step 5: Evaluate the Random Forest Model
def evaluate_random_forest(model, X_test, y_test):
    """Evaluate the Random Forest model and display metrics."""
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nRandom Forest Model Accuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

# Step 6: Build, Train, and Save a Neural Network Model
def build_neural_network(input_shape):
    """Build a Neural Network model."""
    nn_model = Sequential()
    nn_model.add(Dense(128, input_shape=(input_shape,), activation="relu"))
    nn_model.add(Dense(64, activation="relu"))
    nn_model.add(Dense(1, activation="sigmoid"))
    
    nn_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    print("\nNeural Network model built successfully!")
    return nn_model

def train_neural_network(model, X_train, y_train, epochs=50):
    """Train the Neural Network model and save it to disk."""
    print("\nTraining the Neural Network model...")
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_split=0.2, verbose=1)
    
    # Save the model
    model.save("neural_network_model.h5")
    print("\nNeural Network model trained and saved successfully!")
    return history

def evaluate_neural_network(model, X_test, y_test):
    """Evaluate the Neural Network model and display metrics."""
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nNeural Network Model Accuracy: {accuracy:.4f}")

# Step 7: Visualize Training History (Neural Network)
def plot_training_history(history):
    """Plot the training and validation accuracy/loss over epochs."""
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    
    plt.show()

# Step 8: Make Predictions on New Data
def predict_disease(model, new_patient_data, scaler):
    """Predict whether a new patient is diabetic or not."""
    new_patient_data_scaled = scaler.transform([new_patient_data])
    prediction = model.predict(new_patient_data_scaled)
    return "Diabetic" if prediction[0] == 1 else "Not Diabetic"

# Step 9: Interactive Input for New Patient Data
def get_patient_data():
    """Collect patient data interactively from the user."""
    print("\nEnter the following details for the patient:")
    pregnancies = int(input("Number of pregnancies: "))
    glucose = float(input("Glucose level: "))
    blood_pressure = float(input("Blood pressure: "))
    skin_thickness = float(input("Skin thickness: "))
    insulin = float(input("Insulin level: "))
    bmi = float(input("BMI: "))
    diabetes_pedigree = float(input("Diabetes pedigree function: "))
    age = int(input("Age: "))
    
    return [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]

# Main Function
def main():
    # Load and explore the dataset
    filepath = "diabetes.csv"
    try:
        patient_data = load_and_explore_data(filepath)
    except FileNotFoundError as e:
        print(e)
        return

    # Preprocess the data
    X_train, X_test, y_train, y_test, scaler = preprocess_data(patient_data)

    # Train and evaluate the Random Forest model
    rf_model = train_random_forest(X_train, y_train)
    evaluate_random_forest(rf_model, X_test, y_test)

    # Build, train, and evaluate the Neural Network model
    input_shape = X_train.shape[1]
    nn_model = build_neural_network(input_shape)
    history = train_neural_network(nn_model, X_train, y_train, epochs=50)
    evaluate_neural_network(nn_model, X_test, y_test)

    # Visualize training history
    plot_training_history(history)

    # Make predictions on new patient data
    new_patient_data = get_patient_data()
    prediction = predict_disease(rf_model, new_patient_data, scaler)
    print(f"\nPrediction for the patient: {prediction}")

# Run the script
if __name__ == "__main__":
    main()