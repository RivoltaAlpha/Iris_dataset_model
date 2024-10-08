import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Step 1: Load Iris dataset and split it into training and testing sets
iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Initialize and train the model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Step 3: Save the model to disk
joblib.dump(model, 'logistic_regression_model.joblib')
print("Model saved to disk as 'logistic_regression_model.joblib'.")

# Step 4: Load the model from disk
loaded_model = joblib.load('logistic_regression_model.joblib')
print("Model loaded from disk.")

# Step 5: Make predictions with the loaded model
new_data = [[5.1, 3.5, 1.4, 0.2],  # Likely Iris-Setosa
            [6.7, 3.1, 4.7, 1.5]]  # Likely Iris-Versicolor

predictions = loaded_model.predict(new_data)
species = iris.target_names[predictions]
print("Predictions for new data:", species)
