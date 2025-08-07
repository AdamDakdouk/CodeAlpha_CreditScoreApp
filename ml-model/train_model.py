import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
import joblib

def load_and_prepare_data():
    """Load and prepare the dataset"""
    df = pd.read_csv('../data/credit_data.csv')
    
    # Features (X) and target (y)
    feature_columns = ['age', 'income', 'debt_to_income', 'credit_history_length', 
                      'num_credit_accounts', 'payment_history_score', 'credit_utilization', 
                      'num_late_payments', 'employment_years']
    
    X = df[feature_columns]
    y = df['credit_worthy']
    
    return X, y, feature_columns

def train_models(X_train, X_test, y_train, y_test):
    """Train multiple models and compare performance"""
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train model
        if name == 'Random Forest' or name == 'Decision Tree':
            model.fit(X_train, y_train)  # Tree models don't need scaling
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        else:
            model.fit(X_train_scaled, y_train)  # Use scaled data for Logistic Regression
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        results[name] = {
            'model': model,
            'scaler': scaler if name == 'Logistic Regression' else None,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc
        }
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")
    
    return results

def save_best_model(results, feature_columns):
    """Save the best performing model"""
    # Find best model based on F1-score
    best_model_name = max(results.keys(), key=lambda k: results[k]['f1_score'])
    best_model_data = results[best_model_name]
    
    print(f"\nBest model: {best_model_name}")
    print(f"Best F1-Score: {best_model_data['f1_score']:.4f}")
    
    # Save model and scaler
    joblib.dump(best_model_data['model'], 'best_credit_model.pkl')
    if best_model_data['scaler']:
        joblib.dump(best_model_data['scaler'], 'scaler.pkl')
    
    # Save feature names
    joblib.dump(feature_columns, 'feature_columns.pkl')
    
    print("Model saved successfully!")
    return best_model_name, best_model_data

if __name__ == "__main__":
    # Load data
    X, y, feature_columns = load_and_prepare_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Train models
    results = train_models(X_train, X_test, y_train, y_test)
    
    # Save best model
    save_best_model(results, feature_columns)