import pandas as pd
import numpy as np
import os
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def load_data(datapath):
    try:
        return pd.read_csv(datapath)
    except FileNotFoundError:
        print(f"Error: The file {datapath} was not found.")
        raise
    except pd.errors.EmptyDataError:
        print(f"Error: The file {datapath} is empty.")
        raise
    except pd.errors.ParserError:
        print(f"Error: The file {datapath} could not be parsed.")
        raise
    except Exception as e:
        print(f"Unexpected error loading data from {datapath}: {e}")
        raise

def split_data(data):
    try:
        x = data.drop(columns=['class'], axis=1)
        y = data['class']
        return x, y
    except KeyError:
        print("Error: 'class' column not found in the data.")
        raise
    except Exception as e:
        print(f"Unexpected error during data splitting: {e}")
        raise

def load_model(filepath):
    try:
        with open(filepath, "rb") as file:
            model = pickle.load(file)  # Corrected pickle.load instead of pickle.dump
            return model
    except FileNotFoundError:
        print(f"Error: The model file {filepath} was not found.")
        raise
    except pickle.UnpicklingError:
        print(f"Error: The model file {filepath} could not be unpickled.")
        raise
    except Exception as e:
        print(f"Unexpected error loading model from {filepath}: {e}")
        raise

def predict_model(model, x_test, y_test):
    try:
        y_pred = model.predict(x_test)
        acc = accuracy_score(y_test, y_pred)
        pre = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1_scr = f1_score(y_test, y_pred)

        metrics_dict = {
            'acc': acc,
            'precision': pre,
            'recall': recall,
            'f1_score': f1_scr
        }
        return metrics_dict
    except ValueError as e:
        print(f"Error: Incompatible data for model prediction: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error during model prediction: {e}")
        raise

def save_metrics(metrics_dict, filepath):
    try:
        with open(filepath, 'w') as file:  # Use the correct filepath
            json.dump(metrics_dict, file, indent=4)
    except Exception as e:
        print(f"Error saving metrics to {filepath}: {e}")
        raise

def main():
    try:
        data_path = "./data/processed/test_processed.csv"
        model_path = "models/model.pkl"
        metrics_file = "reports/metrics.json"

        test_data = load_data(data_path)
        x_test, y_test = split_data(test_data)
        model = load_model(model_path)
        metrics = predict_model(model, x_test, y_test)
        save_metrics(metrics, metrics_file)  # Save metrics to the correct file path
    except Exception as e:
        raise Exception(f"An error occurred: {e}")

if __name__ == "__main__":  # Correct indentation for main execution block
    main()
