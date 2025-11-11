from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os
from google.cloud import storage

app = Flask(__name__)
CORS(app)

MODEL = None

def load_model():
    storage_client = storage.Client()
    bucket_name_xgb = "smart-dental-t1"
    bucket_name_labid = "smart-dental-lab-id"

    bucket_xgb = storage_client.get_bucket(bucket_name_xgb)
    bucket_labid = storage_client.get_bucket(bucket_name_labid)

    blob_xgb = bucket_xgb.blob("ml_artifacts/xgb_model.joblib")
    blob_labid = bucket_labid.blob("ml_artifacts/lab_encoder.joblib")

    blob_xgb.download_to_filename("xgb_model.joblib")
    blob_labid.download_to_filename("lab_encoder.joblib")

    model = joblib.load("xgb_model.joblib")
    label_encoder = joblib.load("lab_encoder.joblib")

    return {
        "xgb_model": model,
        "label_encoder": label_encoder,
        "X_train_cols": ['POSTAL_CODE', 'REQUIRED_SPEC', 'EXPEDITED', 'REQUIRED_LAB_TYPE', 'LAB_POSTAL_INIT'] 
    }

def predict_top_k_labs(xgb_model, train_label_encoder, X_train_cols, raw_input_data, K=5):

    X_new_raw = pd.DataFrame([raw_input_data])
    X_new = X_new_raw.drop(columns=['RENDERING_NPI'], errors='ignore').copy()

    X_new_processed = X_new.reindex(columns=X_train_cols, fill_value=0)

    probabilities = xgb_model.predict_proba(X_new_processed.values)
    probabilities_for_case = probabilities[0]

    top_k_indices = np.argsort(probabilities_for_case)[-K:][::-1]
    top_k_lab_ids = train_label_encoder.inverse_transform(top_k_indices)
    top_k_probabilities = probabilities_for_case[top_k_indices]
    
    results = [
    {
        'Rank': r,
        'LAB_ID': int(lid),
        'Probability': float(prob),
    }
    for r, lid, prob in zip(range(1, K + 1), top_k_lab_ids, top_k_probabilities)
    ]
    return results

@app.route('/predict', methods=['POST'])
def predict():
    if not request.is_json:
        return jsonify({"error": "Missing JSON in request"}), 400

    raw_input_data = request.get_json()
    
    try:

        MODEL = load_model()
        recommendations = predict_top_k_labs(
            xgb_model=MODEL['xgb_model'],
            train_label_encoder=MODEL['label_encoder'],
            X_train_cols=MODEL['X_train_cols'],
            raw_input_data=raw_input_data,
            K=5
        )

        return jsonify(recommendations)
    
    except Exception as e:
        return jsonify({"error": "Prediction failed", "details": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5051)))