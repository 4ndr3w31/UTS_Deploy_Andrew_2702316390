import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

class RandomForestModel:
    def __init__(self):
        self.model = RandomForestClassifier(max_depth=None, min_samples_leaf=1, min_samples_split=5, n_estimators=200, random_state=42)
        self.scaler = StandardScaler()

    def train(self, X_train, y_train):
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)

    def predict(self, X_test):
        X_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_scaled)

    def save(self, path='best_model_rf.pkl'):
        with open(path, 'wb') as f:
            pickle.dump((self.model, self.scaler), f)

    def load(self, path='best_model_rf.pkl'):
        with open(path, 'rb') as f:
            self.model, self.scaler = pickle.load(f)

# Predict function used in Streamlit app
def predict_new(data: pd.DataFrame):
    try:
        # Load the trained model and scaler
        with open("best_model_rf.pkl", "rb") as f:
            model, scaler = pickle.load(f)
        
        # Scale the data before prediction
        data_scaled = scaler.transform(data)
        
        # Make predictions
        predictions = model.predict(data_scaled)
        return predictions
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None
