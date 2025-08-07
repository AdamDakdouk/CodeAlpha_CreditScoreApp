# 🏦 Credit Score Assessment App

An AI-powered web application that predicts creditworthiness using machine learning algorithms. Built as part of the **CodeAlpha Machine Learning Internship Program**.

![App Demo](https://img.shields.io/badge/Status-Complete-brightgreen) ![ML Model](https://img.shields.io/badge/ML%20Accuracy-91.3%25-blue) ![Tech Stack](https://img.shields.io/badge/Full%20Stack-React%20%7C%20Node.js%20%7C%20Python-orange)

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [ML Model Performance](#ml-model-performance)
- [Project Structure](#project-structure)
- [Installation and Setup](#installation-and-setup)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Model Details](#model-details)

## 🎯 Overview

This full-stack application demonstrates end-to-end machine learning development, from data generation and model training to web deployment. Users input their financial information through a modern, responsive interface and receive instant credit assessments with confidence scores and personalized recommendations.

**⚠️ Disclaimer**: This is an educational project using synthetic data. Results should not be used for actual financial decisions.

## ✨ Features

- **🎨 Modern UI/UX**: Responsive design with glassmorphism effects and smooth animations
- **🤖 AI-Powered Predictions**: Machine learning model with 91%+ accuracy
- **📊 Real-time Analysis**: Instant credit assessment with confidence scoring
- **📝 Interactive Forms**: User-friendly input validation and error handling
- **📈 Visual Results**: Beautiful results page with progress bars and recommendations
- **🔒 Secure Processing**: Data processed securely without storage
- **📱 Mobile Responsive**: Works seamlessly across all devices

## 🛠️ Tech Stack

### Frontend
- **React 18** with TypeScript
- **Tailwind CSS** for styling
- **Axios** for API communication
- **Lucide React** for icons

### Backend
- **Node.js** with Express.js
- **TypeScript** for type safety
- **CORS** for cross-origin requests

### Machine Learning
- **Python 3.13**
- **scikit-learn** for ML algorithms
- **pandas and numpy** for data processing
- **joblib** for model persistence

## 📊 ML Model Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | 91.3% |
| **Precision** | 91.4% |
| **Recall** | 91.2% |
| **F1-Score** | 91.3% |
| **ROC-AUC** | 97.5% |

**Best Model**: Logistic Regression (outperformed Random Forest and Decision Tree)

## 📁 Project Structure

CodeAlpha_CreditScoreApp/
├── ml-model/                 # Machine Learning components
│   ├── ml_env/              # Python virtual environment
│   ├── dataset_generator.py # Synthetic data generation
│   ├── train_model.py       # Model training and evaluation
│   ├── credit_model.py      # Prediction service
│   ├── best_credit_model.pkl # Trained model
│   └── *.pkl               # Model artifacts
├── backend/                 # Node.js API server
│   ├── src/
│   │   └── server.ts       # Express.js server
│   ├── package.json
│   └── tsconfig.json
├── frontend/               # React application
│   ├── src/
│   │   ├── components/     # React components
│   │   │   ├── WelcomePage.tsx
│   │   │   ├── DataInputForm.tsx
│   │   │   └── ResultsPage.tsx
│   │   └── App.tsx
│   ├── public/
│   └── package.json
└── data/                   # Generated datasets
└── credit_data.csv

## 🚀 Installation and Setup

### Prerequisites
- **Node.js** (v16 or higher)
- **Python** (v3.8 or higher)
- **Git**

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/CodeAlpha_CreditScoreApp.git
cd CodeAlpha_CreditScoreApp
```

### 2. Set Up Python Environment and ML Model
```bash
cd ml-model
python -m venv ml_env
```

# Activate virtual environment
# Windows
```bash
ml_env\Scripts\activate
```
# Mac/Linux:
```bash
source ml_env/bin/activate
```

# Install Python dependencies
```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter joblib flask flask-cors
```

# Generate dataset and train model
```bash
python dataset_generator.py
python train_model.py
```

### 3. Set Up Backend
```bash
cd ../backend
npm install
```

### 3. Set Up Frontend
```bash
cd ../frontend
npm install
```

## 🖥️ Usage
### Start Backend Server
```bash
cd backend
npm run dev
```

### Start Frontend Server
```bash
cd frontend
npm start
```

## Using the App

### 1. Welcome Page
Click "Get Started" to begin

### 2.Input Form
Fill in your financial information:
- Age, Income, Debt-to-Income Ratio
- Credit History Length, Number of Credit Accounts
- Payment History Score, Credit Utilization
- Late Payments, Employment Years

### 3. Results
View your credit assessment with:
- Approval/Denial status
- Confidence score
- Detailed recommendations

## Sample Data

### Good Credit Profile
Age: 35, Income: 65000, Debt-to-Income: 0.3
Credit History: 10, Credit Accounts: 5
Payment History: 0.85, Credit Utilization: 0.25
Late Payments: 1, Employment: 8

### Poor Credit Profile
Age: 23, Income: 25000, Debt-to-Income: 1.2
Credit History: 2, Credit Accounts: 12
Payment History: 0.4, Credit Utilization: 0.9
Late Payments: 8, Employment: 1

## Api Endpoints

### Health Check
GET /api/health

### Credit Prediction
POST /api/predict
Content-Type: application/json

{
  "age": 35,
  "income": 65000,
  "debt_to_income": 0.3,
  "credit_history_length": 10,
  "num_credit_accounts": 5,
  "payment_history_score": 0.85,
  "credit_utilization": 0.25,
  "num_late_payments": 1,
  "employment_years": 8
}
