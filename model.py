import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

def load_data(path="starbucks.csv"):
    df = pd.read_csv(path)
    return df

def preprocess_data(df):
    y = df["type"]
    x = df.drop("type", axis=1)

    feature_names = list(x.columns)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    X_train, X_test, y_train, y_test = train_test_split(
        x_scaled, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, scaler, feature_names

def train_model(X_train, y_train):
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    return model
