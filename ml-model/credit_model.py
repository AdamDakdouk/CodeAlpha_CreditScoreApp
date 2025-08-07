import joblib
import pandas as pd
import numpy as np
import json
import sys

class CreditScorePredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model and preprocessing objects"""
        try:
            self.model = joblib.load('best_credit_model.pkl')
            self.feature_columns = joblib.load('feature_columns.pkl')
            
            try:
                self.scaler = joblib.load('scaler.pkl')
            except FileNotFoundError:
                self.scaler = None
                
        except FileNotFoundError as e:
            print(f"Error loading model: {e}")
            raise
    
    def predict_credit_worthiness(self, user_data):
        """Predict credit worthiness for a single user"""
        df = pd.DataFrame([user_data])
        X = df[self.feature_columns]
        
        if self.scaler:
            X_scaled = self.scaler.transform(X)
            prediction = self.model.predict(X_scaled)[0]
            probability = self.model.predict_proba(X_scaled)[0]
        else:
            prediction = self.model.predict(X)[0]
            probability = self.model.predict_proba(X)[0]
        
        result = {
            'credit_worthy': bool(prediction),
            'confidence': float(max(probability)),
            'probability_not_worthy': float(probability[0]),
            'probability_worthy': float(probability[1])
        }
        
        return result

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Called from API with user data
        user_data = json.loads(sys.argv[1])
        predictor = CreditScorePredictor()
        result = predictor.predict_credit_worthiness(user_data)
        print(json.dumps(result))
    else:
        # Test mode
        predictor = CreditScorePredictor()
        test_user = {
            'age': 35,
            'income': 65000,
            'debt_to_income': 0.3,
            'credit_history_length': 10,
            'num_credit_accounts': 5,
            'payment_history_score': 0.85,
            'credit_utilization': 0.25,
            'num_late_payments': 1,
            'employment_years': 8
        }
        result = predictor.predict_credit_worthiness(test_user)
        print("Test prediction:")
        print(f"Credit worthy: {result['credit_worthy']}")
        print(f"Confidence: {result['confidence']:.4f}")