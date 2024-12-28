from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Initialize Flask app
app = Flask(__name__)

# Load and preprocess data
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

# Encoding categorical data
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

sc = StandardScaler()
X = sc.fit_transform(X)

# Build the ANN model
ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Load the model weights
ann.load_weights('model_weights.weights.h5')

@app.route('/')

def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = request.form

        # Extract feature values
        features = [
            int(data['Geography_France']),
            int(data['Geography_Spain']),
            int(data['Geography_Germany']),
            int(data['CreditScore']),
            int(data['Gender']),
            int(data['Age']),
            int(data['Tenure']),
            float(data['Balance']),
            int(data['NumOfProducts']),
            int(data['HasCrCard']),
            int(data['IsActiveMember']),
            float(data['EstimatedSalary'])
        ]

        # Standardize the input using the existing scaler
        features = sc.transform([features])

        # Predict using the ANN model
        prediction = ann.predict(features)
        result = 'Stay' if prediction[0][0] > 0.5 else 'Leave'

        return render_template('index.html', prediction_text=f'Customer will: {result}', probability=f'Probability: {prediction[0][0]:.2f}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
