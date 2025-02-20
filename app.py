from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Initialize Flask app
app = Flask(__name__)

# Load the dataset
data = pd.read_csv("data_stress_3_1_.csv")  # Replace with the correct path

# Prepare the data
X = data.drop(columns=["Stress Levels"])
y = data["Stress Levels"]

# Handle missing values using mean imputation
imputer = SimpleImputer(strategy="mean")
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42, stratify=y)

# Train the Logistic Regression model
logistic_regression = LogisticRegression(random_state=42, max_iter=2000)
logistic_regression.fit(X_train, y_train)

# Evaluate the model
lr_accuracy = accuracy_score(y_test, logistic_regression.predict(X_test))
print(f"Model trained. Accuracy: {lr_accuracy:.2f}")


@app.route('/home', methods=['POST'])
def predict_stress_level():
    """
    Predict stress levels based on physiological parameters.
    """
    try:
        # Get JSON input from the frontend
        user_input = request.json
        
        # Ensure all required inputs are present
        required_fields = ["heart_rate", "respiration_rate", "body_temperature", "hours_of_sleep", "snoring_range"]
        for field in required_fields:
            if field not in user_input:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        # Extract input values
        heart_rate = user_input["heart_rate"]
        respiration_rate = user_input["respiration_rate"]
        body_temperature = user_input["body_temperature"]
        hours_of_sleep = user_input["hours_of_sleep"]
        snoring_range = user_input["snoring_range"]

        # Create a new input array with the required features
        # Assuming the rest of the features are filled with their mean value from the training set
        new_input = np.array([[
            heart_rate, respiration_rate, body_temperature, hours_of_sleep, snoring_range,
            X_train['limb movement'].mean(),  # Default for missing features
            X_train['blood oxygen '].mean(),
            X_train['eye movement'].mean()
        ]])

        # Ensure the input matches the training feature columns
        new_input_df = pd.DataFrame(new_input, columns=X.columns)

        # Predict the stress level
        predicted_stress = logistic_regression.predict(new_input_df)[0]
        print("in stress")

        # Return the predicted stress level
        return jsonify({"predicted_stress_level": predicted_stress})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
