import ray
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

ray.init(address="auto", namespace="training")

@ray.remote
def train_model(path):
    df = pd.read_csv(path)
    X = df.drop("target", axis=1)
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    joblib.dump(model, "/tmp/model.pkl")
    return acc

if __name__ == "__main__":
    future = train_model.remote("/mnt/data/dataset.csv")
    print("Training started...")
    print("Accuracy:", ray.get(future))
