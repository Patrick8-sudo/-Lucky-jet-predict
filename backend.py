from flask import Flask, request, jsonify
import requests
import joblib
import os

app = Flask(__name__)

# Charger le modèle d'apprentissage automatique
model = joblib.load('model.pkl')

# URL de l'API de Lucky Jet
API_URL = "https://api.luckyjet.com/results"

@app.route('/predict', methods=['GET'])
def predict():
    # Récupérer les données de l'API de Lucky Jet
    response = requests.get(API_URL)
    if response.status_code != 200:
        return jsonify({"error": "Failed to retrieve data from Lucky Jet"}), 500
    
    data = response.json()
    
    # Préparer les données pour le modèle (adapter selon votre modèle)
    features = prepare_features(data)
    
    # Prédire le prochain résultat
    prediction = model.predict([features])
    
    return jsonify({"prediction": prediction[0]})

def prepare_features(data):
    # Transformer les données en fonction des caractéristiques du modèle
    # (Adapter cette partie selon votre modèle)
    features = [
        data['some_field_1'],
        data['some_field_2'],
        # Ajouter d'autres champs nécessaires
    ]
    return features

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
