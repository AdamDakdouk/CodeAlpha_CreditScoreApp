const testData = {
  age: 35,
  income: 65000,
  debt_to_income: 0.3,
  credit_history_length: 10,
  num_credit_accounts: 5,
  payment_history_score: 0.85,
  credit_utilization: 0.25,
  num_late_payments: 1,
  employment_years: 8
};

fetch('http://localhost:5000/api/predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(testData)
})
.then(res => res.json())
.then(data => console.log('Success:', data))
.catch(err => console.error('Error:', err));