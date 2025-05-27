"""
 University:	Bellevue University
 Program:		Computer Science
 Course:		CS 440 Artificial Intelligence
 Topic:			Final Project: Supervised Machine Learning
 Desc:			Supports experiments with supervised ML. See the
				Module Lecture instructions and the
				CS440-AI-proj.docx document
 Reading:		Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow 
				textbook, Chapter 2 End-to-End Machine Learning Project
 Developer:		Dr. Rob Flowers
 Version:		v103124
"""

# ................................................................

import sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# Load dataset
def load_data():
    return pd.read_csv("housing.csv")

# Initialize models
def init_model(name):
    models = {
        "logistic": LogisticRegression(max_iter=1000),
        "tree": DecisionTreeClassifier(),
        "svm": SVC(kernel="rbf")  # You added this!
    }
    return models.get(name)

# Show help
def show_help():
    print("""
Usage: py ML.py [option]
Options:
    h       Show housing data statistics
    v       Show data visualizations
    o       Train & test model
    [none]  Show this help screen
""")

# Show statistics
def show_statistics(data):
    print("Descriptive Statistics:\n", data.describe())

# Visualize data
def visualize_data(data):
    data.hist(bins=50, figsize=(12, 8))
    plt.tight_layout()
    plt.show()

# Run model training and prediction
def run_model(data):
    X = data.drop("median_house_value", axis=1)
    y = data["median_house_value"]
    
    # Optional: bin house values into classes for classification
    y = pd.qcut(y, 3, labels=["Low", "Medium", "High"])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = init_model("svm")  # Change model here (svm, logistic, tree)
    if model is None:
        print("Model not found.")
        return

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("\nModel Results:\n")
    print(classification_report(y_test, predictions))

# Main controller
if __name__ == "__main__":
    data = load_data()

    if len(sys.argv) < 2:
        show_help()
    else:
        command = sys.argv[1]

        if command == "h":
            show_statistics(data)
        elif command == "v":
            visualize_data(data)
        elif command == "o":
            run_model(data)
        else:
            show_help()
