import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import tensorflow as tf
import logging
import joblib
import os

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def load_and_explore_data(filepath: str, verbose: bool = True) -> pd.DataFrame:
    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"The file {filepath} does not exist.")
        
        if not filepath.lower().endswith('.csv'):
            raise ValueError("The file must be a CSV file.")
        
        patient_data = pd.read_csv(filepath)
        
        numeric_columns = patient_data.select_dtypes(include=['number']).columns
        patient_data[numeric_columns] = patient_data[numeric_columns].fillna(patient_data[numeric_columns].median())
        
        logging.info("Dataset loaded successfully!")
        
        if verbose:
            logging.info("\nFirst few rows of the dataset:")
            logging.info("\n" + str(patient_data.head()))
        
            logging.info("\nMissing values in the dataset:")
            logging.info("\n" + str(patient_data.isnull().sum()))
        
            logging.info("\nDataset statistics:")
            logging.info("\n" + str(patient_data.describe()))
        
        # Returning the dataset and insights
        insights = {
            "shape": patient_data.shape,
            "missing_values": patient_data.isnull().sum(),
            "statistics": patient_data.describe()
        }

        return patient_data, insights

    except FileNotFoundError as fnf_error:
        logging.error(fnf_error)
    except ValueError as ve:
        logging.error(ve)
    except Exception as e:
        logging.error(f"Unexpected error occurred: {e}")

    return None, None

def validate_data(patient_data):
    if patient_data.isnull().sum().any():
        patient_data.fillna(patient_data.median(), inplace=True)
    if patient_data.duplicated().any():
        patient_data.drop_duplicates(inplace=True)
    return patient_data

def preprocess_data(patient_data):
    """Preprocess the data"""
    
    # Ensure that all feature columns are numeric
    features = patient_data.drop("Outcome", axis=1)
    target = patient_data["Outcome"]
    
    # Attempt to convert features to numeric values
    features = features.apply(pd.to_numeric, errors='coerce')
    
    # Check for any NaN values after conversion
    if features.isnull().any().any():
        logging.error("Non-numeric data detected after conversion!")
        return None, None, None, None, None, None
    
    try:
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    except Exception as e:
        logging.error(f"Error during train_test_split: {e}")
        return None, None, None, None, None, None
    
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = dict(enumerate(class_weights)) 
    
    # Scale the features
    scaler = StandardScaler()
    try:
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    except Exception as e:
        logging.error(f"Error during feature scaling: {e}")
        return None, None, None, None, None, None
    
    print("\nData preprocessing completed!")
    
    return X_train, X_test, y_train, y_test, scaler, class_weights

def balance_dataset(X_train, y_train):
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    return X_train_balanced, y_train_balanced


def tune_random_forest(X_train, y_train):
    """Tune Random Forest hyperparameters using RandomizedSearchCV."""
    
    param_dist = {
        'n_estimators': randint(100, 1000),       
        'max_depth': [None, 10, 20, 30, 40, 50],   
        'min_samples_split': randint(2, 11),         
        'min_samples_leaf': randint(1, 5),          
        'max_features': ['sqrt', 'log2', None],     
        'bootstrap': [True, False],
        'criterion': ['gini', 'entropy']         
    }
    
    rf_model = RandomForestClassifier(random_state=42)
    
    random_search = RandomizedSearchCV(
        estimator=rf_model, 
        param_distributions=param_dist,
        n_iter=150,                                   
        cv=5, 
        verbose=2, 
        random_state=42, 
        n_jobs=-1,
        scoring='accuracy'                           
    )
    
    random_search.fit(X_train, y_train)
    
    print("\nBest Parameters:", random_search.best_params_)
    print("Best Score:", random_search.best_score_)
    
    return random_search.best_estimator_


def cross_validate_model(model, X_train, y_train):
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"Cross-Validation Accuracy: {np.mean(scores):.4f} (± {np.std(scores):.4f})")
    return scores


def select_important_features(model, X_train, threshold=0.01):
    importances = model.feature_importances_
    important_features = X_train.columns[importances > threshold]
    return important_features

def train_xgboost(X_train, y_train, class_weights=None):
    if class_weights:
        scale_pos_weight = class_weights[1] / class_weights[0]
        xgb_model = XGBClassifier(random_state=42, scale_pos_weight=scale_pos_weight)
    else:
        xgb_model = XGBClassifier(random_state=42)
    xgb_model.fit(X_train, y_train)
    return xgb_model


def build_improved_model(input_dim):
    model = Sequential([
        Dense(512, activation='relu', input_shape=(input_dim,), kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.4),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.0001), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    return model


def train_neural_network(
    model: Model, 
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    epochs: int = 50, 
    batch_size: int = 32, 
    validation_split: float = 0.2, 
    model_filename: str = "neural_network_model.h5", 
    early_stopping: bool = False,
    class_weights: dict = None
) -> "History":
    try:
        if X_train.size == 0 or y_train.size == 0:
            raise ValueError("Training data cannot be empty.")

        callbacks = []
        if early_stopping:
            callbacks.append(EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True))

        history = model.fit(
            X_train, y_train, 
            epochs=epochs, 
            batch_size=batch_size, 
            validation_split=validation_split, 
            verbose=1, 
            callbacks=callbacks,
            class_weight=class_weights
        )
        
        model.save(model_filename)
        logging.info(f"Neural Network model trained and saved as {model_filename} successfully!")
        return history

    except ValueError as ve:
        logging.error(f"ValueError: {ve}")
    except Exception as e:
        logging.error(f"Unexpected error occurred: {e}")

    return None





def evaluate_neural_network(
    model: Model, 
    X_test: np.ndarray, 
    y_test: np.ndarray, 
    verbose: int = 0
) -> tuple[float, float] | None:
    """Evaluate the Neural Network model and return metrics."""
    try:
        if X_test.size == 0 or y_test.size == 0:
            raise ValueError("Test dataset (X_test, y_test) cannot be empty.")

        if not hasattr(model, "evaluate"):
            raise AttributeError("The provided model does not have an 'evaluate' method.")

        # Evaluate model
        loss, accuracy = model.evaluate(X_test, y_test, verbose=verbose)
        logging.info(f"Model Accuracy: {accuracy:.2f}")

        return loss, accuracy

    except ValueError as ve:
        logging.error(f"ValueError: {ve}")
    except AttributeError as ae:
        logging.error(f"AttributeError: {ae}")
    except Exception as e:
        logging.error(f"Unexpected error occurred: {e}")

    return None

def plot_training_history(history, metrics=["accuracy", "loss"], figsize=(12, 5)):
    try:
        if not hasattr(history, 'history') or not isinstance(history.history, dict):
            raise ValueError("The provided history object is not valid.")
        plt.figure(figsize=figsize)
        num_plots = len(metrics)
        for i, metric in enumerate(metrics):
            if f"val_{metric}" not in history.history or metric not in history.history:
                logging.warning(f"Missing data for {metric}. Skipping plot.")
                continue
            
            plt.subplot(1, num_plots, i + 1)
            plt.plot(history.history[metric], label=f"Training {metric.capitalize()}")
            plt.plot(history.history[f"val_{metric}"], label=f"Validation {metric.capitalize()}")
            plt.title(f"Training and Validation {metric.capitalize()}")
            plt.xlabel("Epoch")
            plt.ylabel(metric.capitalize())
            plt.legend()
        
        plt.tight_layout()
        plt.show()
        logging.info("Training history plotted successfully.")

    except ValueError as ve:
        logging.error(ve)
    except Exception as e:
        logging.error(f"Unexpected error occurred while plotting: {e}")
        
       
def get_patient_data(feature_names):
    """Collect patient data in the order of feature_names."""
    patient_data = []
    for feature in feature_names:
        value = float(input(f"{feature}: "))
        patient_data.append(value)
    return patient_data

def predict_disease(model, new_patient_data, scaler, feature_names, important_features):
    """Select important features and predict."""
    new_patient_data = np.array(new_patient_data).reshape(1, -1)
    new_patient_scaled = scaler.transform(new_patient_data)
    new_patient_df = pd.DataFrame(new_patient_scaled, columns=feature_names)
    new_patient_selected = new_patient_df[important_features]
   
    if hasattr(model, 'predict_proba'):
        prediction = model.predict_proba(new_patient_selected)[:, 1]
    else:
        prediction = model.predict(new_patient_selected)
    prediction_label = "Diabetic" if prediction > 0.5 else "Not Diabetic"
    return prediction_label


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_test, y_pred))
    
    

def main():
    filepath = "diabetes.csv"
    
    try:
        patient_data, insights = load_and_explore_data(filepath)
        if patient_data is None:
            raise ValueError("Failed to load data")
        logging.info(f"Dataset shape: {patient_data.shape}")
    except Exception as e:
        logging.error(f"Data loading failed: {e}")
        return

    try:
        X_train, X_test, y_train, y_test, scaler, class_weights = preprocess_data(patient_data)
        if X_train is None:
            raise ValueError("Preprocessing failed")
    except Exception as e:
        logging.error(f"Preprocessing error: {e}")
        return

    try:
        X_train_balanced, y_train_balanced = balance_dataset(X_train, y_train)
        class_weights_balanced = compute_class_weight('balanced', 
                                                    classes=np.unique(y_train_balanced), 
                                                    y=y_train_balanced)
        class_weights_balanced = dict(enumerate(class_weights_balanced))
    except Exception as e:
        logging.error(f"Balancing error: {e}")
        return

    # 4. Train/evaluate Random Forest
    try:
        rf_model = tune_random_forest(X_train_balanced, y_train_balanced)
        important_features = select_important_features(rf_model, X_train_balanced)
        X_train_selected = X_train_balanced[important_features]
        X_test_selected = X_test[important_features]
        
        cross_validate_model(rf_model, X_train_selected, y_train_balanced)
        
        evaluate_model(rf_model, X_test_selected, y_test)
    except Exception as e:
        logging.error(f"Random Forest error: {e}")

    try:
        input_shape = X_train_selected.shape[1]
        nn_model = build_improved_model(input_shape)
        history = train_neural_network(
            nn_model, 
            X_train_selected, 
            y_train_balanced,
            epochs=150,
            early_stopping=True,
            class_weights=class_weights_balanced
        )
        
        if history:
            loss, accuracy = evaluate_neural_network(nn_model, X_test_selected, y_test)
            plot_training_history(history)
    except Exception as e:
        logging.error(f"Neural Network error: {e}")

    # 6. XGBoost (assuming train_xgboost is defined)
    try:
        xgb_model = train_xgboost(X_train_selected, y_train_balanced, class_weights_balanced)
        evaluate_model(xgb_model, X_test_selected, y_test)
    except Exception as e:
        logging.error(f"XGBoost error: {e}")

    # 7. Prediction
    try:
        if 'important_features' in locals() and scaler is not None:
            feature_names = patient_data.drop("Outcome", axis=1).columns.tolist()
            selected_feature_names = [feature_names[i] for i in important_features]
            
            new_patient_data = get_patient_data(selected_feature_names)
            
            if new_patient_data:
                prediction = predict_disease(
                    rf_model, 
                    new_patient_data, 
                    scaler,
                    important_features
                )
                logging.info(f"\nPrediction result: {prediction}")
    except Exception as e:
        logging.error(f"Prediction error: {e}")


if __name__ == "__main__":
    main()
    