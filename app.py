from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import json

app = Flask(__name__)

dataset = pd.read_csv("app_test_dashboard_with_prediction")

@app.route('/load_models', methods=['GET'])
def load_model():
    global model
    # Load the model from the file
    with open('model_weights/clf_xgb_o.pkl', 'rb') as f:
        model = joblib.load(f)

    global imputer
    with open('model_weights/imputer.pkl', 'rb') as f:
        imputer = joblib.load(f)

    global colonnes_model
    with open('model_weights/colonnes_attendues.pkl', 'rb') as f:
        colonnes_model = joblib.load(f)

    global scaler
    with open('model_weights/scaler.pkl', 'rb') as f:
        scaler = joblib.load(f)

    return 'Models loaded!'

@app.route("/infos_client", methods=["GET"])
def infos_client():
    id = request.args.get("id_client")

    data_client = dataset[dataset["SK_ID_CURR"] == int(id)]

    print(data_client)
    dict_infos = {
        "status_famille": data_client["NAME_FAMILY_STATUS"].item(),
        "nb_enfant": data_client["CNT_CHILDREN"].item(),
        "age": int(data_client["DAYS_BIRTH"].values / -365),
        "revenus": data_client["AMT_INCOME_TOTAL"].item(),
        "montant_credit": data_client["AMT_CREDIT"].item(),
        "annuites": data_client["AMT_ANNUITY"].item(),
        "montant_bien": data_client["AMT_GOODS_PRICE"].item()
    }

    response = json.loads(data_client.to_json(orient='index'))

    return response

@app.route("/predict_client", methods=["GET"])
def predict_client():
    id = request.args.get("id_client")

    data = dataset[dataset["SK_ID_CURR"] == int(id)]

    # Prépare la requête pour qu'elle soit conforme au modèle
    # ONE HOT ENCODING
    data = pd.get_dummies(data)
    print("Étape 3 réussie.")

    # VALEURS ABERRANTES
    # Create an anomalous flag column
    data['DAYS_EMPLOYED_ANOM'] = data["DAYS_EMPLOYED"] == 365243
    # Replace the anomalous values with nan
    data['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace=True)
    print("Étape 4 réussie.")

    # Traitement des valeurs négatives
    data['DAYS_BIRTH'] = abs(data['DAYS_BIRTH'])
    print("Étape 5 réussie.")

    # CREATION DE VARIABLES
    # CREDIT_INCOME_PERCENT: the percentage of the credit amount relative to a client's income
    # ANNUITY_INCOME_PERCENT: the percentage of the loan annuity relative to a client's income
    # CREDIT_TERM: the length of the payment in months (since the annuity is the monthly amount due
    # DAYS_EMPLOYED_PERCENT: the percentage of the days employed relative to the client's age

    data['CREDIT_INCOME_PERCENT'] = data['AMT_CREDIT'] / data['AMT_INCOME_TOTAL']
    data['ANNUITY_INCOME_PERCENT'] = data['AMT_ANNUITY'] / data[
        'AMT_INCOME_TOTAL']
    data['CREDIT_TERM'] = data['AMT_ANNUITY'] / data['AMT_CREDIT']
    data['DAYS_EMPLOYED_PERCENT'] = data['DAYS_EMPLOYED'] / data['DAYS_BIRTH']
    print("Étape 6 réussie.")

    # Récupère les colonnes attendues par le modèle
    colonnes_attendues = colonnes_model
    print(colonnes_attendues)
    print("Étape 7 réussie.")

    # Identify the columns that are missing from the received dataframe
    missing_columns = set(colonnes_attendues) - set(data.columns)
    print("Étape 8 réussie.")

    # Add the missing columns to the received dataframe with a default value
    for col in missing_columns:
        data[col] = 0
    print("Étape 9 réussie.")

    # Reorder the columns to match the order of the expected columns
    data = data[colonnes_attendues]
    data = data.reindex(columns=colonnes_attendues)
    print("Étape 10 réussie.")

    # Transform the data using the imputer and scaler
    data = data[imputer.feature_names_in_]
    data = imputer.transform(data)
    print("Étape 11 réussie.")
    data = scaler.transform(data)
    print("Étape 12 réussie.")

    # Make a prediction using the model
    prediction = model.predict(data)
    prediction_proba = model.predict_proba(data)
    print("Étape 13 réussie.")

    # Convert the prediction to a list
    prediction = prediction.tolist()
    prediction_proba = prediction_proba.tolist()
    print("Étape 14 réussie.")

    # Return the prediction as a response
    return jsonify({'prediction': prediction}, {'prediction_proba': prediction_proba})


@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the request
    data = request.get_json()
    print("Étape 1 réussie.")

    # Convert the received data to a dataframe
    data = pd.DataFrame(data, index=[0])
    print("Étape 2 réussie.")

    # Prépare la requête pour qu'elle soit conforme au modèle
    # ONE HOT ENCODING
    data = pd.get_dummies(data)
    print("Étape 3 réussie.")

    # VALEURS ABERRANTES
    # Create an anomalous flag column
    data['DAYS_EMPLOYED_ANOM'] = data["DAYS_EMPLOYED"] == 365243
    # Replace the anomalous values with nan
    data['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace=True)
    print("Étape 4 réussie.")

    # Traitement des valeurs négatives
    data['DAYS_BIRTH'] = abs(data['DAYS_BIRTH'])
    print("Étape 5 réussie.")

    # CREATION DE VARIABLES
    # CREDIT_INCOME_PERCENT: the percentage of the credit amount relative to a client's income
    # ANNUITY_INCOME_PERCENT: the percentage of the loan annuity relative to a client's income
    # CREDIT_TERM: the length of the payment in months (since the annuity is the monthly amount due
    # DAYS_EMPLOYED_PERCENT: the percentage of the days employed relative to the client's age

    data['CREDIT_INCOME_PERCENT'] = data['AMT_CREDIT'] / data['AMT_INCOME_TOTAL']
    data['ANNUITY_INCOME_PERCENT'] = data['AMT_ANNUITY'] / data[
        'AMT_INCOME_TOTAL']
    data['CREDIT_TERM'] = data['AMT_ANNUITY'] / data['AMT_CREDIT']
    data['DAYS_EMPLOYED_PERCENT'] = data['DAYS_EMPLOYED'] / data['DAYS_BIRTH']
    print("Étape 6 réussie.")

    # Récupère les colonnes attendues par le modèle
    colonnes_attendues = colonnes_model
    print(colonnes_attendues)
    print("Étape 7 réussie.")

    # Identify the columns that are missing from the received dataframe
    missing_columns = set(colonnes_attendues) - set(data.columns)
    print("Étape 8 réussie.")

    # Add the missing columns to the received dataframe with a default value
    for col in missing_columns:
        data[col] = 0
    print("Étape 9 réussie.")

    # Reorder the columns to match the order of the expected columns
    data = data[colonnes_attendues]
    data = data.reindex(columns=colonnes_attendues)
    print("Étape 10 réussie.")

    # Transform the data using the imputer and scaler
    data = data[imputer.feature_names_in_]
    data = imputer.transform(data)
    print("Étape 11 réussie.")
    data = scaler.transform(data)
    print("Étape 12 réussie.")

    # Make a prediction using the model
    prediction = model.predict(data)
    prediction_proba = model.predict_proba(data)
    print("Étape 13 réussie.")

    # Convert the prediction to a list
    prediction = prediction.tolist()
    prediction_proba = prediction_proba.tolist()
    print("Étape 14 réussie.")

    # Return the prediction as a response
    return jsonify({'prediction': prediction}, {'prediction_proba': prediction_proba})


if __name__ == '__main__':
    app.run()
