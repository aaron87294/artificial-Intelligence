import sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

def load_data():
    return pd.read_csv("housing.csv")

def init_model(name):
    models = {
        "logistic": LogisticRegression(max_iter=1000),
        "tree": DecisionTreeClassifier(),
        "svm": SVC(kernel="rbf")
    }
    return models.get(name)

def show_help():
    print("""
Usage: python3 ML.py [option]
Options:
    h       Show housing data statistics
    v       Show data visualizations
    o       Train & test model
    [none]  Show this help screen
""")

def show_statistics(data):
    print("Descriptive Statistics:\n", data.describe())

def visualize_data(data):
    import matplotlib.pyplot as plt

    # Drop non-numeric columns for histogram plotting
    numeric_data = data.select_dtypes(include=['number'])

    numeric_data.hist(bins=50, figsize=(12, 8))
    plt.tight_layout()
    plt.show()

def run_model(data):
    from sklearn.preprocessing import LabelEncoder
    data = data.copy()
    data = data.dropna()


    # Encode all object (non-numeric) columns
    for column in data.select_dtypes(include=['object']).columns:
        data[column] = LabelEncoder().fit_transform(data[column])

    X = data.drop("median_house_value", axis=1)
    y = data["median_house_value"]
    y = pd.qcut(y, 3, labels=["Low", "Medium", "High"])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = init_model("svm")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("\nModel Results:\n")
    print(classification_report(y_test, predictions))

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
