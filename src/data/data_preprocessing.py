import pandas as pd 
import numpy as np
from sklearn.preprocessing import LabelEncoder 
import os 

def load_data(filepath):
    try: 
        return pd.read_csv(filepath)
    except Exception as e:
        print(f"Error loading data from {filepath}: {e}")
        raise
#train_data=pd.read_csv("./data/raw/train.csv")
#test_data = pd.read_csv("./data/raw/test.csv")
def encode_categorical_columns(data):
    try:
        if data.select_dtypes(include=['object']).shape[1] > 0:
            label_encoder = LabelEncoder()
            categorical_columns = data.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                data[col] = label_encoder.fit_transform(data[col])
        return data
    except Exception as e:
        print(f"Error encoding categorical columns: {e}")
        raise

def save_data(data,filepath):
    try:
        data.to_csv(filepath,index=False)
    except Exception as e:
        print(f"Error saving data to {filepath}: {e}")
        raise

def main():
    try:
        data_raw_path = "./data/raw"
        data_processed_path = "./data/processed"

        train_data=load_data(os.path.join(data_raw_path,"train.csv"))
        test_data= load_data(os.path.join(data_raw_path,"test.csv"))
    

        train_processed_data = encode_categorical_columns(train_data)
        test_processed_data = encode_categorical_columns(test_data)

#data_path = os.path.join("data","processed")

        os.makedirs(data_processed_path)

        save_data(train_data,os.path.join(data_processed_path,"train_processed.csv"))
        save_data(test_data, os.path.join(data_processed_path,"test_processed.csv"))
    except Exception as e:
        print(f"Error in main processing: {e}")

if __name__ == "__main__":
    main()

#train_processed_data.to_csv(os.path.join(data_path,"train_processed.csv"),index=False)
#test_processed_data.to_csv(os.path.join(data_path,"test_processed.csv"),index=False)