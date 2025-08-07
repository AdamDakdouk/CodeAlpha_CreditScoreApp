import pandas as pd
import numpy as np

def generate_credit_dataset(n_samples=5000):
    """Generate synthetic credit scoring dataset with realistic correlations"""
    np.random.seed(42)
    
    # Generate base features
    age = np.random.randint(18, 80, n_samples)
    income = np.random.normal(50000, 20000, n_samples).clip(15000, 150000)
    employment_years = np.minimum(age - 18, np.random.randint(0, 40, n_samples))
    credit_history_length = np.minimum(age - 18, np.random.randint(0, 30, n_samples))
    
    # Make DTI and credit utilization correlated
    base_debt_ratio = np.random.uniform(0, 1.2, n_samples)
    debt_to_income = base_debt_ratio + np.random.normal(0, 0.1, n_samples)
    debt_to_income = debt_to_income.clip(0, 1.5)
    
    credit_utilization = base_debt_ratio * 0.7 + np.random.uniform(0, 0.3, n_samples)
    credit_utilization = credit_utilization.clip(0, 1)
    
    # Other features
    num_credit_accounts = np.random.randint(1, 15, n_samples)  # At least 1 account
    payment_history_score = np.random.beta(3, 1.5, n_samples)  # Better distribution
    num_late_payments = np.random.poisson(1.5, n_samples)  # Fewer late payments
    
    data = {
        'age': age,
        'income': income,
        'debt_to_income': debt_to_income,
        'credit_history_length': credit_history_length,
        'num_credit_accounts': num_credit_accounts,
        'payment_history_score': payment_history_score,
        'credit_utilization': credit_utilization,
        'num_late_payments': num_late_payments,
        'employment_years': employment_years
    }
    
    df = pd.DataFrame(data)
    
    # Simplified but working credit score calculation
    credit_score = (
        400 +  # Base score
        (df['income'] / 1000) * 1.5 +  # Income factor (up to ~225 points)
        (1 - df['debt_to_income']) * 50 +  # Debt ratio (up to 50 points)
        df['credit_history_length'] * 4 +  # History (up to 120 points)
        df['payment_history_score'] * 100 +  # Payment history (up to 100 points)
        (1 - df['credit_utilization']) * 30 +  # Utilization (up to 30 points)
        np.maximum(0, (5 - df['num_late_payments'])) * 8 +  # Late payment penalty
        np.random.normal(0, 20, n_samples)  # Random variation
    ).clip(300, 850)
    
    df['credit_score'] = credit_score.round().astype(int)
    df['credit_worthy'] = (df['credit_score'] >= 650).astype(int)
    
    return df

if __name__ == "__main__":
    dataset = generate_credit_dataset()
    dataset.to_csv('../data/credit_data.csv', index=False)
    print("Dataset created with", len(dataset), "samples")
    print("Target distribution:", dataset['credit_worthy'].value_counts().to_dict())
    print("Credit score range:", dataset['credit_score'].min(), "to", dataset['credit_score'].max())