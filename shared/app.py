from flask import Flask, request, jsonify
import pandas as pd
import os
from utility import process_all_groups_Prophet,process_all_groups

app = Flask(__name__)

RESULT_DIR = "shared/result/"

def train_predict(train, test, ids):
    """
    Placeholder for the actual train and predict logic.
    """
    # Filter data for specific IDs
    train_filtered = train[train['default_rank'].isin(ids)]
    test_filtered = test[test['default_rank'].isin(ids)]

    # Example: Predict sales (replace with actual logic)
    # predictions = test_filtered.copy()
    # predictions['sales_forecast'] = train_filtered['sales'].mean()  # Example computation

    print('stated_predicting')
    predictions=process_all_groups_Prophet(train_filtered, test_filtered, ids, save_interval=1, save_path="predicted_prophet_data_v2")

    return predictions

@app.route('/')
def home():
    return "Hello, World!"

@app.route('/process', methods=['POST','GET'])
def process_request():
    """
    Handle requests to process data for a subset of IDs.
    """

    
    data = request.json
    ids = data['ids']
    train_file = data['train_file']
    test_file = data['test_file']

    # Read CSVs
    print(train_file)
    print("satish")
    print(os.listdir(os.curdir))
    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)

    # Run predictions
    results = train_predict(train, test, ids)

    # Save results
    result_file = os.path.join(RESULT_DIR, f"results_{min(ids)}_{max(ids)}.csv")
    results.to_csv(result_file, index=False)

    return jsonify({"result_file": result_file}), 200

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8082)  # Adjust port for each server
